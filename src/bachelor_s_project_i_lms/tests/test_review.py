import csv
import json

import pytest

from bachelor_s_project_i_lms.graph.review import (
    apply_edits,
    export_mermaid,
    export_nodes_csv,
    run,
    write_template,
)


def edge(u, v, conf=0.8, origin="local"):
    return {"from_temp_id": u, "to_temp_id": v, "confidence": conf,
            "rationale": "test edge", "origin": origin}


def atom(tid, title, root="r", needs_review=False):
    return {"temp_id": tid, "title": title, "summary": "", "depth": 1, "root": root,
            "parent_temp_id": "p", "is_atomic": True, "needs_review": needs_review,
            "learning_objective": f"The learner can explain {title}.",
            "estimated_time_min": 8, "difficulty_weight": 2, "type": "concept",
            "source_chunk_ids": ["chunk_p1_2"]}


# ----------------------------- export -----------------------------

def test_nodes_csv_has_degrees_and_neighbours(tmp_path):
    atoms = [atom("a", "Alpha"), atom("b", "Beta"), atom("c", "Gamma")]
    edges = [edge("a", "b"), edge("b", "c")]
    path = export_nodes_csv(atoms, edges, tmp_path / "nodes.csv")

    rows = {r["temp_id"]: r for r in csv.DictReader(path.open(encoding="utf-8"))}
    assert rows["a"]["out_degree"] == "1" and rows["a"]["in_degree"] == "0"
    assert rows["b"]["prerequisites"] == "Alpha"
    assert rows["b"]["dependents"] == "Gamma"
    assert rows["a"]["isolated"] == "False"


def test_nodes_csv_marks_isolated_and_review(tmp_path):
    atoms = [atom("a", "Alpha", needs_review=True), atom("b", "Beta")]
    path = export_nodes_csv(atoms, [], tmp_path / "nodes.csv")
    rows = {r["temp_id"]: r for r in csv.DictReader(path.open(encoding="utf-8"))}
    assert rows["a"]["isolated"] == "True"
    assert rows["a"]["needs_review"] == "True"


def test_mermaid_groups_roots_and_labels_edges(tmp_path):
    atoms = [atom("a", "Alpha", root="Ch 1"), atom("b", "Beta", root="Ch 2")]
    edges = [edge("a", "b", 0.75, "cross_branch")]
    text = export_mermaid(atoms, edges, tmp_path / "graph.mmd").read_text(encoding="utf-8")

    assert text.startswith("flowchart TD")
    assert 'subgraph sg0["Ch 1"]' in text
    assert 'subgraph sg1["Ch 2"]' in text
    assert 'a -->|"0.75 cross_branch"| b' in text


def test_mermaid_flags_review_and_isolated(tmp_path):
    atoms = [atom("a", "Alpha", needs_review=True), atom("b", "Beta")]
    text = export_mermaid(atoms, [], tmp_path / "graph.mmd").read_text(encoding="utf-8")
    assert "class a needsReview" in text
    assert "class a,b isolated" in text


def test_template_lists_flagged_atoms_and_is_not_overwritten(tmp_path):
    path = tmp_path / "review_edits.json"
    write_template([atom("a", "Alpha", needs_review=True), atom("b", "Beta")], path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert list(data["node_overrides"]) == ["a"]
    assert data["manual_edges"] == []

    path.write_text('{"node_overrides": {}, "manual_edges": [], "removed_edges": []}', encoding="utf-8")
    write_template([atom("a", "Alpha", needs_review=True)], path)
    assert json.loads(path.read_text(encoding="utf-8"))["node_overrides"] == {}


# ----------------------------- apply edits -----------------------------

def test_apply_node_override():
    atoms = [atom("a", "Alpha", needs_review=True)]
    out_atoms, _, report = apply_edits(
        atoms, [], {"node_overrides": {"a": {"needs_review": False, "estimated_time_min": 5}}}
    )
    assert out_atoms[0]["needs_review"] is False
    assert out_atoms[0]["estimated_time_min"] == 5
    assert report["errors"] == []


def test_apply_rejects_unknown_field_and_id():
    atoms = [atom("a", "Alpha")]
    _, _, report = apply_edits(
        atoms, [],
        {"node_overrides": {"a": {"bogus": 1}, "zzz": {"needs_review": False}}},
    )
    assert len(report["errors"]) == 2


def test_duplicate_temp_id_keeps_first_occurrence():
    first = atom("a", "Alpha")
    second = {**atom("a", "Alpha"), "parent_temp_id": "other"}
    out_atoms, _, report = apply_edits([first, second], [], {})
    assert len(out_atoms) == 1
    assert out_atoms[0]["parent_temp_id"] == "p"
    assert report["duplicate_temp_ids_dropped"] == ["a"]


def test_merge_rewires_edges_and_dedupes():
    atoms = [atom("a", "Alpha"), atom("b", "Beta"), atom("c", "Gamma"), atom("d", "Delta")]
    edges = [edge("a", "c"), edge("b", "c"), edge("c", "d")]
    out_atoms, out_edges, report = apply_edits(
        atoms, edges, {"merge_nodes": [{"keep": "a", "drop": "b", "reason": "duplicate"}]}
    )
    assert {a["temp_id"] for a in out_atoms} == {"a", "c", "d"}
    pairs = {(e["from_temp_id"], e["to_temp_id"]) for e in out_edges}
    assert pairs == {("a", "c"), ("c", "d")}
    assert report["merged"][0]["kept_title"] == "Alpha"


def test_merge_rejects_unknown_and_self():
    atoms = [atom("a", "Alpha"), atom("b", "Beta")]
    _, _, report = apply_edits(atoms, [], {"merge_nodes": [
        {"keep": "a", "drop": "zzz"},
        {"keep": "a", "drop": "a"},
    ]})
    assert len(report["errors"]) == 2


def test_remove_node_drops_incident_edges():
    atoms = [atom("a", "Alpha"), atom("b", "Beta"), atom("c", "Gamma")]
    edges = [edge("a", "b"), edge("b", "c")]
    out_atoms, out_edges, report = apply_edits(
        atoms, edges, {"remove_nodes": ["b"]}
    )
    assert {a["temp_id"] for a in out_atoms} == {"a", "c"}
    assert out_edges == []
    assert report["nodes_removed"] == [{"temp_id": "b", "title": "Beta"}]


def test_remove_node_rejects_unknown():
    _, _, report = apply_edits([atom("a", "Alpha")], [], {"remove_nodes": ["zzz"]})
    assert report["errors"] == ["remove_nodes: unknown temp_id 'zzz'"]


def test_manual_edge_is_added_with_manual_origin():
    atoms = [atom("a", "Alpha"), atom("b", "Beta")]
    _, edges, report = apply_edits(
        atoms, [], {"manual_edges": [{"from_temp_id": "a", "to_temp_id": "b", "rationale": "missing"}]}
    )
    assert len(edges) == 1
    assert edges[0]["origin"] == "manual"
    assert edges[0]["confidence"] == 1.0
    assert report["manual_edges_added"][0]["from"] == "Alpha"


def test_manual_edge_rejects_self_loop_duplicate_and_unknown():
    atoms = [atom("a", "Alpha"), atom("b", "Beta")]
    existing = [edge("a", "b")]
    _, edges, report = apply_edits(
        atoms, existing,
        {"manual_edges": [
            {"from_temp_id": "a", "to_temp_id": "a"},
            {"from_temp_id": "a", "to_temp_id": "b"},
            {"from_temp_id": "a", "to_temp_id": "zzz"},
        ]},
    )
    assert len(edges) == 1
    assert len(report["errors"]) == 3


def test_removed_edge():
    atoms = [atom("a", "Alpha"), atom("b", "Beta")]
    _, edges, report = apply_edits(
        atoms, [edge("a", "b")],
        {"removed_edges": [{"from_temp_id": "a", "to_temp_id": "b"}]},
    )
    assert edges == []
    assert len(report["edges_removed"]) == 1


def test_cycle_from_edit_drops_weakest_edge():
    atoms = [atom("a", "Alpha"), atom("b", "Beta")]
    _, edges, report = apply_edits(
        atoms, [edge("a", "b")],
        {"manual_edges": [{"from_temp_id": "b", "to_temp_id": "a"}]},
    )
    assert report["cycles"] == []
    assert len(report["cycle_edges_removed"]) == 1
    assert report["cycle_edges_removed"][0]["from"] == "Alpha"
    assert {(e["from_temp_id"], e["to_temp_id"]) for e in edges} == {("b", "a")}


# ----------------------------- entry point -----------------------------

def test_run_writes_exports_and_reviewed_files(tmp_path):
    atoms = [atom("a", "Alpha", needs_review=True), atom("b", "Beta")]
    atoms_path = tmp_path / "atoms.json"
    atoms_path.write_text(json.dumps(atoms), encoding="utf-8")
    edges_path = tmp_path / "final_edges.json"
    edges_path.write_text(json.dumps([edge("a", "b")]), encoding="utf-8")

    report = run(
        atoms_path=atoms_path,
        edges_path=edges_path,
        nodes_csv=tmp_path / "nodes.csv",
        mermaid_path=tmp_path / "graph.mmd",
        edits_path=tmp_path / "review_edits.json",
        reviewed_atoms_path=tmp_path / "reviewed_atoms.json",
        reviewed_edges_path=tmp_path / "reviewed_edges.json",
        report_path=tmp_path / "review_report.json",
    )
    assert report["n_atoms"] == 2 and report["n_edges"] == 1
    assert (tmp_path / "nodes.csv").exists()
    assert (tmp_path / "graph.mmd").exists()
    assert json.loads((tmp_path / "reviewed_edges.json").read_text(encoding="utf-8"))[0]["origin"] == "local"


def test_run_refuses_invalid_edits(tmp_path):
    atoms = [atom("a", "Alpha"), atom("b", "Beta")]
    atoms_path = tmp_path / "atoms.json"
    atoms_path.write_text(json.dumps(atoms), encoding="utf-8")
    edges_path = tmp_path / "final_edges.json"
    edges_path.write_text(json.dumps([edge("a", "b")]), encoding="utf-8")
    edits_path = tmp_path / "review_edits.json"
    edits_path.write_text(json.dumps({
        "node_overrides": {},
        "manual_edges": [{"from_temp_id": "b", "to_temp_id": "zzz"}],
        "removed_edges": [],
    }), encoding="utf-8")

    with pytest.raises(RuntimeError):
        run(
            atoms_path=atoms_path,
            edges_path=edges_path,
            nodes_csv=tmp_path / "nodes.csv",
            mermaid_path=tmp_path / "graph.mmd",
            edits_path=edits_path,
            reviewed_atoms_path=tmp_path / "reviewed_atoms.json",
            reviewed_edges_path=tmp_path / "reviewed_edges.json",
            report_path=tmp_path / "review_report.json",
        )
    assert not (tmp_path / "reviewed_atoms.json").exists()

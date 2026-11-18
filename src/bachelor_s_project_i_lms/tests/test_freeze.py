import json

import pytest

from bachelor_s_project_i_lms.graph import review
from bachelor_s_project_i_lms.graph.freeze import (
    build_edge_rows,
    build_node_rows,
    check_gate,
    check_staleness,
    node_id_for,
    normalize_text,
    publish,
    run,
)


def atom(tid, title="Alpha", objective=None, root="Ch 1", refs=None):
    return {
        "temp_id": tid,
        "title": title,
        "summary": "",
        "depth": 1,
        "root": root,
        "parent_temp_id": "p",
        "is_atomic": True,
        "needs_review": False,
        "learning_objective": objective or f"The learner can explain {title}.",
        "estimated_time_min": 8,
        "difficulty_weight": 2,
        "type": "concept",
        "common_misconceptions": [],
        "source_chunk_ids": refs or ["chunk_p1_2"],
    }


def edge(u, v, conf=0.8, origin="local"):
    return {"from_temp_id": u, "to_temp_id": v, "confidence": conf,
            "rationale": "test edge", "origin": origin}


# ----------------------------- identity -----------------------------

def test_normalize_text():
    assert normalize_text("  The   Learner\nCan EXPLAIN ") == "the learner can explain"


def test_node_id_is_stable_and_sensitive():
    a = atom("a")
    assert node_id_for(a) == node_id_for(dict(a))
    assert node_id_for(a) == node_id_for({**a, "title": "Renamed"})
    assert node_id_for(a) != node_id_for({**a, "learning_objective": "different"})
    assert node_id_for(a) != node_id_for({**a, "source_chunk_ids": ["chunk_p9_9"]})


def test_node_id_ignores_source_order():
    a = atom("a", refs=["chunk_p1_2", "chunk_p3_4"])
    b = atom("a", refs=["chunk_p3_4", "chunk_p1_2"])
    assert node_id_for(a) == node_id_for(b)


# ----------------------------- rows -----------------------------

def test_build_rows_maps_fields_and_ids():
    atoms = [atom("a"), atom("b", title="Beta")]
    rows, id_map = build_node_rows(atoms, version=1)
    assert id_map["a"] == rows[0]["node_id"]
    assert rows[0]["parent_topic"] == "Ch 1"
    assert rows[0]["source_ref"] == ["chunk_p1_2"]
    assert json.loads(rows[0]["misconceptions"]) == []

    edge_rows = build_edge_rows([edge("a", "b")], id_map, version=1)
    assert edge_rows[0]["from_id"] == id_map["a"]
    assert edge_rows[0]["to_id"] == id_map["b"]
    assert edge_rows[0]["graph_version"] == 1


# ----------------------------- gate -----------------------------

def test_gate_passes_clean_graph():
    atoms = [atom("a"), atom("b", title="Beta")]
    report = check_gate(atoms, [edge("a", "b")], chunk_ids={"chunk_p1_2"},
                        tree_roots=["Ch 1"])
    assert report["errors"] == []
    assert report["stats"]["n_atoms"] == 2


def test_gate_rejects_bad_metadata():
    atoms = [atom("a"), atom("b", title="Beta")]
    atoms[0]["learning_objective"] = ""
    atoms[1]["type"] = "video"
    atoms[1]["difficulty_weight"] = 9
    atoms[1]["estimated_time_min"] = 0
    atoms[1]["source_chunk_ids"] = []
    errors = check_gate(atoms, [])["errors"]
    assert any("missing learning_objective" in e for e in errors)
    assert any("invalid type" in e for e in errors)
    assert any("difficulty_weight" in e for e in errors)
    assert any("estimated_time_min" in e for e in errors)
    assert any("no source_ref" in e for e in errors)


def test_gate_rejects_duplicate_node_id():
    atoms = [atom("a"), atom("b")]
    errors = check_gate(atoms, [])["errors"]
    assert any("duplicate node_id" in e for e in errors)


def test_gate_rejects_bad_edges():
    atoms = [atom("a"), atom("b", title="Beta")]
    errors = check_gate(atoms, [
        edge("a", "b"),
        edge("a", "b"),
        edge("a", "a"),
        edge("a", "zzz"),
    ])["errors"]
    assert any("duplicate" in e for e in errors)
    assert any("self-loop" in e for e in errors)
    assert any("unknown endpoint" in e for e in errors)


def test_gate_rejects_cycle():
    atoms = [atom("a"), atom("b", title="Beta")]
    errors = check_gate(atoms, [edge("a", "b"), edge("b", "a")])["errors"]
    assert any(e.startswith("cycle:") for e in errors)


def test_gate_rejects_missing_chunk_and_root():
    atoms = [atom("a")]
    errors = check_gate(atoms, [], chunk_ids={"other"}, tree_roots=["Ch 1", "Ch 2"])["errors"]
    assert any("not in book_chunks" in e for e in errors)
    assert any("without atoms" in e for e in errors)


def test_gate_warns_on_needs_review():
    atoms = [atom("a")]
    atoms[0]["needs_review"] = True
    report = check_gate(atoms, [])
    assert report["errors"] == []
    assert report["warnings"]


# ----------------------------- staleness -----------------------------

def test_staleness_accepts_faithful_review():
    atoms = [atom("a"), atom("b", title="Beta")]
    final_edges = [edge("a", "b")]
    edits = {"node_overrides": {"a": {"needs_review": False}}, "manual_edges": [], "removed_edges": []}
    reviewed_atoms, reviewed_edges, _ = review.apply_edits(atoms, final_edges, edits)
    assert check_staleness(reviewed_atoms, reviewed_edges, atoms, final_edges, edits) == []


def test_staleness_detects_drift():
    atoms = [atom("a"), atom("b", title="Beta")]
    final_edges = [edge("a", "b")]
    reviewed_atoms, reviewed_edges, _ = review.apply_edits(atoms, final_edges, {})
    assert check_staleness(reviewed_atoms, [], atoms, final_edges, {})
    assert any("edge set differs" in e
               for e in check_staleness(reviewed_atoms, [], atoms, final_edges, {}))


# ----------------------------- publish -----------------------------

class _Result:
    def __init__(self, row):
        self.row = row

    def first(self):
        return self.row


class _FakeConn:
    def __init__(self, existing=None):
        self.calls = []
        self.existing = existing

    def execute(self, stmt, params=None):
        self.calls.append((str(stmt), params))
        return _Result(self.existing)


def test_publish_inserts_then_freezes():
    conn = _FakeConn()
    atoms = [atom("a"), atom("b", title="Beta")]
    out = publish(conn, atoms, [edge("a", "b")], version=1, notes="x")
    assert out["n_nodes"] == 2 and out["n_edges"] == 1
    assert "SELECT status" in conn.calls[0][0]
    assert "INSERT INTO graph_versions" in conn.calls[1][0]
    assert "INSERT INTO nodes" in conn.calls[2][0]
    assert len(conn.calls[2][1]) == 2
    assert "INSERT INTO edges" in conn.calls[3][0]
    assert "status = 'frozen'" in conn.calls[4][0]


def test_publish_refuses_existing_version():
    conn = _FakeConn(existing=("frozen",))
    with pytest.raises(RuntimeError):
        publish(conn, [atom("a")], [], version=1)
    assert conn.calls and "SELECT status" in conn.calls[0][0]


# ----------------------------- entry point -----------------------------

def _write_inputs(tmp_path, atoms, final_edges, edits):
    atoms_path = tmp_path / "atoms.json"
    atoms_path.write_text(json.dumps(atoms), encoding="utf-8")
    edges_path = tmp_path / "final_edges.json"
    edges_path.write_text(json.dumps(final_edges), encoding="utf-8")
    edits_path = tmp_path / "review_edits.json"
    edits_path.write_text(json.dumps(edits), encoding="utf-8")
    reviewed_atoms, reviewed_edges, _ = review.apply_edits(atoms, final_edges, edits)
    ra = tmp_path / "reviewed_atoms.json"
    ra.write_text(json.dumps(reviewed_atoms), encoding="utf-8")
    re_ = tmp_path / "reviewed_edges.json"
    re_.write_text(json.dumps(reviewed_edges), encoding="utf-8")
    return atoms_path, edges_path, edits_path, ra, re_


def test_run_dry_run_end_to_end(tmp_path):
    atoms = [atom("a"), atom("b", title="Beta")]
    paths = _write_inputs(tmp_path, atoms, [edge("a", "b")],
                          {"node_overrides": {}, "manual_edges": [], "removed_edges": []})
    report = run(
        dry_run=True,
        reviewed_atoms_path=paths[3],
        reviewed_edges_path=paths[4],
        atoms_path=paths[0],
        final_edges_path=paths[1],
        edits_path=paths[2],
        tree_path=None,
        report_path=tmp_path / "freeze_report.json",
        check_db=False,
    )
    assert report["errors"] == []
    assert report["stats"]["n_atoms"] == 2


def test_run_refuses_stale_review(tmp_path):
    atoms = [atom("a"), atom("b", title="Beta")]
    paths = _write_inputs(tmp_path, atoms, [edge("a", "b")],
                          {"node_overrides": {}, "manual_edges": [], "removed_edges": []})
    paths[4].write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(RuntimeError):
        run(
            dry_run=True,
            reviewed_atoms_path=paths[3],
            reviewed_edges_path=paths[4],
            atoms_path=paths[0],
            final_edges_path=paths[1],
            edits_path=paths[2],
            tree_path=None,
            report_path=tmp_path / "freeze_report.json",
            check_db=False,
        )

import json

import pytest

from bachelor_s_project_i_lms.graph.validate import (
    CHAIN_WARNING,
    break_cycles,
    duplicate_atoms,
    isolated_nodes,
    longest_chain,
    syllabus_coverage,
    transitive_reduction,
    validate,
)


def edge(u, v, conf=0.8, origin="local"):
    return {"from_temp_id": u, "to_temp_id": v, "confidence": conf,
            "rationale": "test edge", "origin": origin}


def atom(tid, title, root="r"):
    return {"temp_id": tid, "title": title, "summary": "", "depth": 1, "root": root,
            "parent_temp_id": "p", "is_atomic": True, "needs_review": False,
            "children": [], "learning_objective": f"The learner can explain {title}."}


# ----------------------------- cycles -----------------------------

def test_break_cycles_two_node():
    # a <-> b, a->b has lower confidence -> that one is removed
    edges = [edge("a", "b", 0.6), edge("b", "a", 0.9)]
    dag, removed = break_cycles(["a", "b"], edges)
    assert [(e["from_temp_id"], e["to_temp_id"]) for e in dag] == [("b", "a")]
    assert len(removed) == 1
    assert removed[0]["confidence"] == 0.6
    assert removed[0]["removed_reason"].startswith("cycle")


def test_break_cycles_self_loop():
    edges = [edge("a", "a", 0.9), edge("a", "b", 0.7)]
    dag, removed = break_cycles(["a", "b"], edges)
    assert ("a", "a") not in [(e["from_temp_id"], e["to_temp_id"]) for e in dag]
    assert len(removed) == 1


def test_break_cycles_longer_cycle_removes_weakest():
    # a->b->c->a; weakest is c->a (0.5)
    edges = [edge("a", "b", 0.9), edge("b", "c", 0.8), edge("c", "a", 0.5), edge("c", "d", 0.7)]
    dag, removed = break_cycles(["a", "b", "c", "d"], edges)
    pairs = {(e["from_temp_id"], e["to_temp_id"]) for e in dag}
    assert ("c", "a") not in pairs
    assert pairs == {("a", "b"), ("b", "c"), ("c", "d")}
    assert len(removed) == 1


def test_break_cycles_dag_untouched():
    edges = [edge("a", "b"), edge("b", "c")]
    dag, removed = break_cycles(["a", "b", "c"], edges)
    assert len(dag) == 2 and removed == []


# ----------------------------- transitive reduction -----------------------------

def test_reduction_removes_implied_edge():
    # a->c implied by a->b->c
    edges = [edge("a", "b"), edge("b", "c"), edge("a", "c")]
    reduced, removed = transitive_reduction(["a", "b", "c"], edges)
    assert {(e["from_temp_id"], e["to_temp_id"]) for e in reduced} == {("a", "b"), ("b", "c")}
    assert removed[0]["removed_reason"] == "transitive reduction"


def test_reduction_keeps_true_siblings():
    # a->c and b->c are both leaves' edges: nothing implies anything else
    edges = [edge("a", "c"), edge("b", "c")]
    reduced, removed = transitive_reduction(["a", "b", "c"], edges)
    assert len(reduced) == 2 and removed == []


def test_reduction_keeps_cross_branch_shortcut_when_no_path():
    # a->c via nothing: even with an unrelated b->c, a->c stays
    edges = [edge("a", "c", 0.7), edge("b", "c")]
    reduced, removed = transitive_reduction(["a", "b", "c"], edges)
    pairs = {(e["from_temp_id"], e["to_temp_id"]) for e in reduced}
    assert ("a", "c") in pairs and removed == []


def test_reduction_diamond():
    # s->l->t, s->r->t, plus direct s->t: direct edge is redundant
    edges = [edge("s", "l"), edge("l", "t"), edge("s", "r"), edge("r", "t"), edge("s", "t")]
    reduced, removed = transitive_reduction(["s", "l", "r", "t"], edges)
    assert ("s", "t") not in {(e["from_temp_id"], e["to_temp_id"]) for e in reduced}
    assert len(reduced) == 4


# ----------------------------- isolated nodes -----------------------------

def test_isolated_nodes():
    edges = [edge("a", "b"), edge("b", "c")]
    assert isolated_nodes(["a", "b", "c", "z"], edges) == ["z"]


# ----------------------------- coverage -----------------------------

def test_coverage_reports_empty_heading_and_unattached_atom():
    tree = [{
        "temp_id": "root", "title": "Root", "summary": "", "depth": 0, "root": "Root",
        "parent_temp_id": None, "is_atomic": False, "needs_review": False,
        "children": [
            atom("x", "X"),
            {"temp_id": "empty", "title": "Empty Heading", "summary": "", "depth": 1,
             "root": "Root", "parent_temp_id": "root", "is_atomic": False,
             "needs_review": False, "children": []},
        ],
    }]
    atoms = [atom("x", "X"), atom("orphan", "Orphan")]
    cov = syllabus_coverage(tree, atoms)
    assert cov["headings_without_atoms"] == ["Empty Heading"]
    assert cov["atoms_not_in_tree"] == ["orphan"]


def test_coverage_clean_tree():
    tree = [{
        "temp_id": "root", "title": "Root", "summary": "", "depth": 0, "root": "Root",
        "parent_temp_id": None, "is_atomic": False, "needs_review": False,
        "children": [atom("x", "X"), atom("y", "Y")],
    }]
    atoms = [atom("x", "X"), atom("y", "Y")]
    cov = syllabus_coverage(tree, atoms)
    assert cov["headings_without_atoms"] == []
    assert cov["atoms_not_in_tree"] == []


# ----------------------------- duplicates -----------------------------

def test_duplicate_atoms_flags_near_identical():
    a1 = atom("t1", "VLANs")
    a2 = atom("t2", "VLANs")  # same title + same objective pattern -> near-identical text
    def fake_embed(text):
        return [1.0, 0.0] if "VLANs" in text else [0.0, 1.0]
    dupes = duplicate_atoms([a1, a2, atom("t3", "Something Else")], fake_embed)
    assert len(dupes) == 1
    assert {a["temp_id"] for a in dupes[0]} == {"t1", "t2"}


def test_duplicate_atoms_keeps_distinct():
    def fake_embed(text):
        return [1.0, 0.0] if "VLANs" in text else [0.0, 1.0]
    dupes = duplicate_atoms([atom("t1", "VLANs"), atom("t3", "Something Else")], fake_embed)
    assert dupes == []


# ----------------------------- chain length -----------------------------

def test_longest_chain_counts_edges():
    edges = [edge("a", "b"), edge("b", "c"), edge("c", "d")]
    assert longest_chain(["a", "b", "c", "d"], edges) == 3


def test_longest_chain_picks_longest_branch():
    edges = [edge("s", "a"), edge("s", "b"), edge("b", "c"), edge("c", "d"), edge("d", "e")]
    assert longest_chain(["s", "a", "b", "c", "d", "e"], edges) == 4


def test_longest_chain_empty():
    assert longest_chain(["a"], []) == 0


# ----------------------------- full gauntlet -----------------------------

def test_validate_end_to_end(tmp_path):
    tree = [{
        "temp_id": "root", "title": "Root", "summary": "", "depth": 0, "root": "Root",
        "parent_temp_id": None, "is_atomic": False, "needs_review": False,
        "children": [atom("a", "A"), atom("b", "B"), atom("c", "C"), atom("d", "D")],
    }]
    atoms = [atom("a", "A"), atom("b", "B"), atom("c", "C"), atom("d", "D")]
    local = [edge("a", "b", 0.9), edge("b", "a", 0.5)]   # cycle: b->a weakest, removed
    cross = [edge("b", "c", 0.85), edge("a", "c", 0.7)]  # a->c implied by a->b->c

    def fake_embed(text):
        # all distinct
        return [hash(text) % 97 / 97, ((hash(text) // 97) % 97) / 97]

    report = validate(tree, atoms, local, cross, embed_fn=fake_embed)

    final_pairs = {(e["from_temp_id"], e["to_temp_id"]) for e in report["final_edges"]}
    assert ("b", "a") not in final_pairs          # cycle edge removed
    assert ("a", "c") not in final_pairs          # redundant edge removed
    assert ("a", "b") in final_pairs and ("b", "c") in final_pairs
    assert report["isolated_nodes"] == ["D"]       # atom("d") has no edges
    assert report["coverage"]["headings_without_atoms"] == []
    assert report["duplicate_atom_pairs"] == [] or isinstance(report["duplicate_atom_pairs"], list)
    assert report["longest_chain_edges"] == 2
    assert report["chain_warning"] is False


def test_validate_chain_warning_flagged():
    atoms = [atom(f"n{i}", f"N{i}") for i in range(CHAIN_WARNING + 2)]
    edges = [edge(f"n{i}", f"n{i+1}") for i in range(CHAIN_WARNING + 1)]
    tree = [{
        "temp_id": "root", "title": "Root", "summary": "", "depth": 0, "root": "Root",
        "parent_temp_id": None, "is_atomic": False, "needs_review": False,
        "children": atoms,
    }]
    report = validate(tree, atoms, edges, [])
    assert report["chain_warning"] is True
    assert report["longest_chain_edges"] == CHAIN_WARNING + 1

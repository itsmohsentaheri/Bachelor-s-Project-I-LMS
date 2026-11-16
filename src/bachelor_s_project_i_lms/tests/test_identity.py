"""Identity tests: a node's temp_id must encode its position in the tree, so the
same topic under two different parents is two nodes, not one aliased node.

Regressions the old (root, title) hash allowed:
- two distinct atoms collapsing to one temp_id,
- self-loops in edges.json once both copies received sibling edges,
- aliased edges (an edge asserted about copy A landing on copy B).
"""
from bachelor_s_project_i_lms.graph.decompose import _new_node, _tid


def test_same_title_different_parents_get_different_ids():
    """The Link Layer chapter emitted 'VLANs' at depth 1 AND under its
    'Ethernet and Link-Layer Switches' node: those must be two ids."""
    chapter_root = _tid("The Link Layer and LANs", "root", "The Link Layer and LANs")
    chapter = _new_node("The Link Layer and LANs", "", None, 0, "The Link Layer and LANs")
    ethernet = _new_node("Ethernet and Link-Layer Switches", "", chapter["temp_id"], 1, "The Link Layer and LANs")

    flat = _new_node("Virtual LANs (VLANs)", "", chapter["temp_id"], 1, "The Link Layer and LANs")
    nested = _new_node("Virtual LANs (VLANs)", "", ethernet["temp_id"], 2, "The Link Layer and LANs")

    assert flat["temp_id"] != nested["temp_id"]


def test_same_title_same_parent_gets_same_id():
    """Deterministic: identical placement still yields one id (resume relies on this)."""
    parent = _new_node("P", "", None, 0, "R")
    a = _new_node("Child", "", parent["temp_id"], 1, "R")
    b = _new_node("Child", "", parent["temp_id"], 1, "R")
    assert a["temp_id"] == b["temp_id"]


def test_root_id_scheme_unchanged():
    """Roots keep the legacy (root, title) hash so existing checkpoints resume."""
    root = _new_node("Transport Layer", "", None, 0, "Transport Layer")
    assert root["temp_id"] == _tid("Transport Layer", "Transport Layer")


def test_id_not_reusable_across_chapters():
    """Same title under a same-named node in a different chapter is distinct."""
    r1 = _new_node("R", "", None, 0, "Chapter One")
    r2 = _new_node("R", "", None, 0, "Chapter Two")
    assert r1["temp_id"] != r2["temp_id"]

import json

from bachelor_s_project_i_lms.graph.prereqs import (
    _cache_key,
    _dedupe,
    _sinks,
    _sources,
    leaf_edges,
    local_prereqs,
    run,
    save_edges,
)


def node(title, children=None, is_atomic=False, temp_id=None, summary=""):
    return {
        "temp_id": temp_id or f"id_{title}",
        "title": title,
        "summary": summary,
        "depth": 1,
        "root": "root",
        "parent_temp_id": None,
        "is_atomic": is_atomic,
        "needs_review": False,
        "children": children or [],
    }


def edges_json(*edges):
    """Build the raw JSON a local_prereq call would return."""
    return json.dumps(
        {"edges": [
            {"prerequisite": p, "dependent": d, "confidence": c, "rationale": "test rationale text"}
            for (p, d, c) in edges
        ]}
    )


# ----------------------------- LLM + cache -----------------------------

def test_local_prereqs_cached(tmp_path, monkeypatch):
    tree = node("Parent", [node("A", is_atomic=True), node("B", is_atomic=True)])
    calls = []

    def fake_generate(messages, system):
        calls.append(messages)
        return edges_json(("A", "B", 0.9)), {}

    # redirect cache into tmp so the test never touches real files
    monkeypatch.setattr("bachelor_s_project_i_lms.graph.prereqs.CACHE_DIR", tmp_path)
    out = local_prereqs(tree, generate=fake_generate)
    assert len(out.edges) == 1
    assert len(calls) == 1

    # second call must hit the cache, not the LLM
    out2 = local_prereqs(tree, generate=fake_generate)
    assert len(calls) == 1
    assert out2 == out


def test_local_prereqs_cache_key_covers_children():
    a = node("P", [node("A"), node("B")])
    b = node("P", [node("A"), node("C")])
    # same title, different children -> different cache file
    assert _cache_key(a).name != _cache_key(b).name


# ----------------------------- sinks / sources -----------------------------

def simple_chain(a, b, c):
    """Three atomic leaves with edges a->b, b->c."""
    return [
        {"from_temp_id": a, "to_temp_id": b, "confidence": 0.9, "rationale": "r", "origin": "local"},
        {"from_temp_id": b, "to_temp_id": c, "confidence": 0.9, "rationale": "r", "origin": "local"},
    ]


def test_sinks_and_sources_of_chain():
    n = node("Topic", [node("A", is_atomic=True), node("B", is_atomic=True), node("C", is_atomic=True)])
    edges = simple_chain("id_A", "id_B", "id_C")
    sinks = [l["title"] for l in _sinks(n, edges)]
    sources = [l["title"] for l in _sources(n, edges)]
    assert sinks == ["C"]           # nothing depends on C
    assert sources == ["A"]         # A has no internal prerequisite


# ----------------------------- transformation -----------------------------

def test_atomic_to_atomic_edge_kept_as_is(tmp_path, monkeypatch):
    tree = node("Parent", [
        node("A", is_atomic=True),
        node("B", is_atomic=True),
    ])

    def fake_generate(messages, system):
        return edges_json(("A", "B", 0.85)), {}

    monkeypatch.setattr("bachelor_s_project_i_lms.graph.prereqs.CACHE_DIR", tmp_path)
    edges = leaf_edges(tree, generate=fake_generate)
    assert len(edges) == 1
    assert edges[0]["from_temp_id"] == "id_A"
    assert edges[0]["to_temp_id"] == "id_B"
    assert edges[0]["confidence"] == 0.85
    assert "expanded from sibling edge" not in edges[0]["rationale"]


def test_compound_edge_expanded_to_leaves(tmp_path, monkeypatch):
    """A (compound, internal chain a1->a2) is prereq of B (compound, chain b1->b2).
    The edge must become a2 -> b1 (A's sink to B's source)."""
    a = node("A", [
        node("a1", is_atomic=True),
        node("a2", is_atomic=True),
    ])
    b = node("B", [
        node("b1", is_atomic=True),
        node("b2", is_atomic=True),
    ])
    tree = node("Parent", [a, b])

    def fake_generate(messages, system):
        content = messages[0]["content"]
        if "a1" in content:  # internal call: A's children
            return edges_json(("a1", "a2", 0.9)), {}
        if "b1" in content:  # internal call: B's children
            return edges_json(("b1", "b2", 0.9)), {}
        return edges_json(("A", "B", 0.8)), {}  # sibling call: A -> B

    monkeypatch.setattr("bachelor_s_project_i_lms.graph.prereqs.CACHE_DIR", tmp_path)
    edges = leaf_edges(tree, generate=fake_generate)
    pairs = {(e["from_temp_id"], e["to_temp_id"]) for e in edges}
    # internal chains + the transformed cross-sibling edge (A's sink -> B's source)
    assert pairs == {("id_a1", "id_a2"), ("id_b1", "id_b2"), ("id_a2", "id_b1")}
    transformed = [e for e in edges if (e["from_temp_id"], e["to_temp_id"]) == ("id_a2", "id_b1")]
    assert transformed[0]["confidence"] == 0.8
    assert "expanded from sibling edge" in transformed[0]["rationale"]


def test_compound_to_atomic_and_atomic_to_compound(tmp_path, monkeypatch):
    """Mixed case: compound -> atomic expands sinks to that single atom;
    atomic -> compound connects to the atom's sources."""
    a = node("A", [node("a1", is_atomic=True), node("a2", is_atomic=True)])  # no internal edges
    b = node("B", is_atomic=True)
    tree = node("Parent", [a, b])

    def fake_generate(messages, system):
        content = messages[0]["content"]
        if "a1" in content:  # internal call for A's children
            return edges_json(), {}
        return edges_json(("A", "B", 0.7)), {}

    monkeypatch.setattr("bachelor_s_project_i_lms.graph.prereqs.CACHE_DIR", tmp_path)
    edges = leaf_edges(tree, generate=fake_generate)
    pairs = {(e["from_temp_id"], e["to_temp_id"]) for e in edges}
    # A has no internal edges, so ALL its leaves are sinks -> both connect to B
    assert pairs == {("id_a1", "id_B"), ("id_a2", "id_B")}


def test_unknown_sibling_titles_dropped(tmp_path, monkeypatch):
    tree = node("Parent", [node("A", is_atomic=True), node("B", is_atomic=True)])

    def fake_generate(messages, system):
        # "C" does not exist among the children — must be dropped, not crash
        return edges_json(("C", "B", 0.9)), {}

    monkeypatch.setattr("bachelor_s_project_i_lms.graph.prereqs.CACHE_DIR", tmp_path)
    edges = leaf_edges(tree, generate=fake_generate)
    assert edges == []


def test_dedupe_keeps_highest_confidence():
    es = [
        {"from_temp_id": "x", "to_temp_id": "y", "confidence": 0.6, "rationale": "r1", "origin": "local"},
        {"from_temp_id": "x", "to_temp_id": "y", "confidence": 0.9, "rationale": "r2", "origin": "local"},
        {"from_temp_id": "y", "to_temp_id": "z", "confidence": 0.8, "rationale": "r3", "origin": "local"},
    ]
    deduped = _dedupe(es)
    assert len(deduped) == 2
    assert deduped[0]["confidence"] == 0.9


# ----------------------------- entry point -----------------------------

def test_run_uses_tree_checkpoint(tmp_path, monkeypatch):
    tree = [node("Root", [node("A", is_atomic=True), node("B", is_atomic=True)])]
    tree_path = tmp_path / "tree.json"
    tree_path.write_text(json.dumps(tree))

    def fake_generate(messages, system):
        return edges_json(("A", "B", 0.9)), {}

    monkeypatch.setattr("bachelor_s_project_i_lms.graph.prereqs.CACHE_DIR", tmp_path)
    edges_path = tmp_path / "edges.json"
    edges = run(generate=fake_generate, tree_path=tree_path, edges_path=edges_path)

    assert [(e["from_temp_id"], e["to_temp_id"]) for e in edges] == [("id_A", "id_B")]
    saved = json.loads(edges_path.read_text())
    assert saved == edges


def test_save_edges_roundtrip(tmp_path):
    es = [{"from_temp_id": "x", "to_temp_id": "y", "confidence": 0.9, "rationale": "r", "origin": "local"}]
    save_edges(es, tmp_path / "edges.json")
    assert json.loads((tmp_path / "edges.json").read_text()) == es

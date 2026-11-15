import json

import pytest

from bachelor_s_project_i_lms.graph.cross_branch import (
    CONFIDENCE_THRESHOLD,
    TOP_K,
    _cache_key,
    _min_page,
    candidates,
    confidence_histogram,
    judge_pair,
    run,
    save_cross_edges,
    load_cross_edges,
    verdicts_to_edges,
)
from bachelor_s_project_i_lms.llm.schemas import CrossBranchPairVerdict, DirectionalVerdict


def atom(title, temp_id, root, chunks=(), objective=None):
    return {
        "temp_id": temp_id,
        "title": title,
        "summary": "",
        "depth": 2,
        "root": root,
        "parent_temp_id": "id_root",
        "is_atomic": True,
        "needs_review": False,
        "children": [],
        "learning_objective": objective or f"The learner can explain {title}.",
        "source_chunk_ids": list(chunks),
    }


def verdict(a_before=False, a_conf=0.9, b_before=False, b_conf=0.9):
    return CrossBranchPairVerdict(
        a_before_b=DirectionalVerdict(
            is_prerequisite=a_before, confidence=a_conf,
            rationale="A is needed to understand B." if a_before else "A is not needed for B.",
        ),
        b_before_a=DirectionalVerdict(
            is_prerequisite=b_before, confidence=b_conf,
            rationale="B is needed to understand A." if b_before else "B is not needed for A.",
        ),
    )


def verdict_json(a_before=False, a_conf=0.9, b_before=False, b_conf=0.9):
    v = verdict(a_before, a_conf, b_before, b_conf)
    return v.model_dump_json(), {}


# ----------------------------- book position -----------------------------

def test_min_page_parses_chunk_ids():
    a = atom("Delay", "t1", "r1", chunks=["chunk_p48_52", "chunk_p60_63"])
    assert _min_page(a) == 48


def test_min_page_none_when_unparseable():
    a = atom("Delay", "t1", "r1", chunks=[])
    assert _min_page(a) is None


# ----------------------------- candidates -----------------------------

def test_candidates_cross_root_only():
    a1 = atom("HTTP", "a1", "Application Layer")
    a2 = atom("FTP", "a2", "Application Layer")  # same root: never a candidate
    b1 = atom("TCP", "b1", "Transport Layer")
    vectors = {"a1": [1.0, 0.0], "a2": [0.9, 0.1], "b1": [0.95, 0.05]}
    pairs = candidates([a1, a2, b1], vectors)
    # cross-root pairs only: a1<->b1 AND a2<->b1 (the cap is per proposing atom)
    assert pairs == [("a1", "b1"), ("a2", "b1")]


def test_proposals_respect_top_k_per_atom():
    from bachelor_s_project_i_lms.graph.cross_branch import _proposals_for_atom

    # one atom faces 12 atoms of another root: it proposes at most top_k of them
    r1 = [atom("X", "a1", "r1")]
    r2 = [atom(f"Y{i}", f"b{i}", "r2") for i in range(12)]
    vectors = {a["temp_id"]: [1.0, float(i) / 12] for i, a in enumerate([*r1, *r2])}
    proposals = _proposals_for_atom(r1[0], [*r1, *r2], vectors, TOP_K)
    assert len(proposals) == TOP_K
    assert all(p[0] == "a1" for p in proposals)


def test_candidates_chunk_overlap_adds_missed_pair():
    # orthogonal vectors: similarity will not rank b1 in a1's top-k
    a1 = atom("Channel partitioning", "a1", "Link Layer", chunks=["chunk_p520_525"])
    filler = [atom(f"F{i}", f"f{i}", "Filler Root") for i in range(10)]
    b1 = atom("Multiple access protocols", "b1", "Transport Layer", chunks=["chunk_p520_525"])
    vectors = {
        "a1": [1.0, 0.0],
        "b1": [0.0, 1.0],  # cosine 0 with a1
        **{f"f{i}": [1.0, 0.01 * (i + 1)] for i in range(10)},  # all more similar to a1 than b1 is
    }
    pairs = candidates([a1, b1, *filler], vectors)
    assert ("a1", "b1") in pairs  # found via shared chunk despite zero similarity


def test_candidates_deterministic():
    r1 = [atom("X", "a1", "r1"), atom("Y", "a2", "r1")]
    r2 = [atom("Z", "b1", "r2"), atom("W", "b2", "r2")]
    vectors = {"a1": [1.0, 0.0], "a2": [0.8, 0.6], "b1": [0.9, 0.1], "b2": [0.1, 0.9]}
    assert candidates([*r1, *r2], vectors) == candidates([*r2, *r1], vectors)


# ----------------------------- cache -----------------------------

def test_judge_pair_cached(tmp_path, monkeypatch):
    monkeypatch.setattr("bachelor_s_project_i_lms.graph.cross_branch.CACHE_DIR", tmp_path)
    calls = []

    def fake_generate(messages, system):
        calls.append(messages)
        return verdict_json(a_before=True)

    a, b = atom("Sockets", "a1", "r1"), atom("HTTP", "b1", "r2")
    judge_pair(a, b, generate=fake_generate)
    judge_pair(a, b, generate=fake_generate)
    assert len(calls) == 1  # second call hit the cache


def test_cache_key_covers_objective_change():
    a1 = atom("Sockets", "a1", "r1", objective="The learner can write a socket.")
    a2 = atom("Sockets", "a1", "r1", objective="The learner can describe socket options.")
    b = atom("HTTP", "b1", "r2")
    assert _cache_key(a1, b).name != _cache_key(a2, b).name


# ----------------------------- verdict -> edges -----------------------------

def test_verdicts_to_edges_threshold():
    a, b = atom("Sockets", "a1", "r1"), atom("HTTP", "b1", "r2")
    # a_before_b true but 0.6 < 0.7 -> dropped; b_before_a false -> no edge
    out = verdicts_to_edges(a, b, verdict(a_before=True, a_conf=0.6))
    assert out == []


def test_verdicts_to_edges_keeps_direction():
    a, b = atom("Sockets", "a1", "r1", chunks=["chunk_p100_105"]), atom("HTTP", "b1", "r2", chunks=["chunk_p250_260"])
    out = verdicts_to_edges(a, b, verdict(a_before=True, a_conf=0.85))
    assert len(out) == 1
    e = out[0]
    assert e["from_temp_id"] == "a1" and e["to_temp_id"] == "b1"
    assert e["confidence"] == 0.85
    assert e["origin"] == "cross_branch"
    assert "backward" not in e  # forward edge (p100 -> p250)


def test_backward_edge_flagged():
    # from-page 250 > to-page 100: kept edge points backward in the book
    late = atom("Congestion control", "late", "Transport Layer", chunks=["chunk_p300_310"])
    early = atom("Packet delay", "early", "Internet", chunks=["chunk_p48_55"])
    out = verdicts_to_edges(late, early, verdict(a_before=True, a_conf=0.85))
    assert len(out) == 1
    assert out[0].get("backward") is True


def test_no_edge_when_neither_direction():
    a, b = atom("FTP", "a1", "r1"), atom("DNS", "b1", "r2")
    assert verdicts_to_edges(a, b, verdict()) == []


# ----------------------------- histogram -----------------------------

def test_histogram_buckets():
    vs = {
        ("a", "b"): verdict(a_conf=0.2, b_conf=0.4),
        ("c", "d"): verdict(a_conf=0.6, b_conf=0.75),
        ("e", "f"): verdict(a_conf=0.95, b_conf=0.7),
    }
    hist = confidence_histogram(vs)
    assert hist == {"0.0-0.3": 1, "0.3-0.5": 1, "0.5-0.7": 1, "0.7-0.9": 2, "0.9-1.0": 1}


# ----------------------------- entry point -----------------------------

def test_run_end_to_end(tmp_path, monkeypatch):
    atoms = [
        atom("Sockets", "a1", "Application Layer", chunks=["chunk_p100_105"]),
        atom("HTTP", "a2", "Application Layer"),
        atom("TCP", "b1", "Transport Layer", chunks=["chunk_p100_105"]),
    ]
    atoms_path = tmp_path / "atoms.json"
    atoms_path.write_text(json.dumps(atoms))

    def fake_embed(text):
        # a1 and b1 embed identically -> similar; a2 orthogonal
        return [1.0, 0.0] if "Sockets" in text or "TCP" in text else [0.0, 1.0]

    def fake_generate(messages, system):
        user = messages[0]["content"]
        if user.startswith("Atom A\nTitle: Sockets"):
            return verdict_json(a_before=True, a_conf=0.9)  # sockets -> TCP
        return verdict_json()  # everything else: no relation

    monkeypatch.setattr("bachelor_s_project_i_lms.graph.cross_branch.CACHE_DIR", tmp_path)
    edges_path = tmp_path / "cross_edges.json"
    edges, hist = run(
        generate=fake_generate,
        embed_fn=fake_embed,
        atoms_path=atoms_path,
        edges_path=edges_path,
    )
    assert [(e["from_temp_id"], e["to_temp_id"]) for e in edges] == [("a1", "b1")]
    assert edges[0]["origin"] == "cross_branch"
    saved = json.loads(edges_path.read_text())
    assert len(saved) == 1
    # histogram counts every judged direction (2 pairs x 2 directions)
    assert sum(hist.values()) == 4


def test_run_resumes_without_recalling_llm(tmp_path, monkeypatch):
    atoms = [
        atom("Sockets", "a1", "Application Layer"),
        atom("TCP", "b1", "Transport Layer"),
    ]
    atoms_path = tmp_path / "atoms.json"
    atoms_path.write_text(json.dumps(atoms))

    calls = []

    def fake_generate(messages, system):
        calls.append(messages)
        return verdict_json(a_before=True, a_conf=0.9)

    def fake_embed(text):
        return [1.0, 0.0]

    monkeypatch.setattr("bachelor_s_project_i_lms.graph.cross_branch.CACHE_DIR", tmp_path)
    edges_path = tmp_path / "cross_edges.json"
    run(generate=fake_generate, embed_fn=fake_embed, atoms_path=atoms_path, edges_path=edges_path)
    n_first = len(calls)
    run(generate=fake_generate, embed_fn=fake_embed, atoms_path=atoms_path, edges_path=edges_path)
    assert len(calls) == n_first  # cache: second run made zero new LLM calls


def test_cross_edges_roundtrip(tmp_path):
    edges = [{"from_temp_id": "a", "to_temp_id": "b", "confidence": 0.9, "rationale": "r", "origin": "cross_branch"}]
    save_cross_edges(edges, tmp_path / "cross_edges.json")
    assert load_cross_edges(tmp_path / "cross_edges.json") == edges


def test_run_requires_atoms_checkpoint(tmp_path):
    with pytest.raises(RuntimeError, match="atom_meta"):
        run(embed_fn=lambda t: [1.0], atoms_path=tmp_path / "missing.json")

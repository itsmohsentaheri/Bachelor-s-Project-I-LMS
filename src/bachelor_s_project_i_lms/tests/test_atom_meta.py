import json

from bachelor_s_project_i_lms.graph.atom_meta import (
    _cache_key,
    atom_meta,
    load_atoms,
    run,
    save_atoms,
)


def atom(title, temp_id=None):
    return {
        "temp_id": temp_id or f"id_{title}",
        "title": title,
        "summary": "",
        "depth": 2,
        "root": "root",
        "parent_temp_id": "id_root",
        "is_atomic": True,
        "needs_review": False,
        "children": [],
    }


def chunks_json(*chunk_ids):
    return [
        {"chunk_id": cid, "page": 10, "text": f"text for {cid}"} for cid in chunk_ids
    ]


def meta_json(valid_ids, invalid_ids=()):
    """Build raw JSON an atom_meta call would return."""
    misconceptions = [
        {"id": "m1", "description": "thinks the concept works differently"},
        {"id": "m2", "description": "confuses it with a related concept"},
    ]
    return json.dumps(
        {
            "learning_objective": "The learner can explain this concept precisely.",
            "estimated_time_min": 5,
            "difficulty_weight": 3,
            "type": "concept",
            "common_misconceptions": misconceptions,
            "source_chunk_ids": [*valid_ids, *invalid_ids],
        }
    )


def fake_fetch(*chunk_ids):
    def f(query):
        return chunks_json(*chunk_ids)
    return f


# ----------------------------- provenance -----------------------------

def test_invalid_chunk_ids_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("bachelor_s_project_i_lms.graph.atom_meta.CACHE_DIR", tmp_path)
    a = atom("Routing")
    # model cites c1 (real), c9 (invented): only c1 must survive
    out = atom_meta(a, generate=lambda m, s: (meta_json(["c1"], ["c9"]), {}),
                    fetch=fake_fetch("c1", "c2"))
    assert out.source_chunk_ids == ["c1"]
    assert a["source_chunk_ids"] == ["c1"]
    assert a["needs_review"] is False


def test_no_valid_chunks_flags_needs_review(tmp_path, monkeypatch):
    monkeypatch.setattr("bachelor_s_project_i_lms.graph.atom_meta.CACHE_DIR", tmp_path)
    a = atom("Routing")
    # model invents every chunk id: nothing survives -> atom flagged, no metadata attached
    out = atom_meta(a, generate=lambda m, s: (meta_json([], ["c9"]), {}),
                    fetch=fake_fetch("c1"))
    assert out is None
    assert a["needs_review"] is True
    assert "learning_objective" not in a


# ----------------------------- cache -----------------------------

def test_atom_meta_cached(tmp_path, monkeypatch):
    monkeypatch.setattr("bachelor_s_project_i_lms.graph.atom_meta.CACHE_DIR", tmp_path)
    calls = []

    def fake_generate(messages, system):
        calls.append(messages)
        return meta_json(["c1"]), {}

    a = atom("Routing")
    fetch = fake_fetch("c1", "c2")
    out = atom_meta(a, generate=fake_generate, fetch=fetch)
    assert len(calls) == 1

    a2 = atom("Routing")
    out2 = atom_meta(a2, generate=fake_generate, fetch=fetch)
    assert len(calls) == 1  # cache hit, no second API call
    assert out2 == out


def test_cache_key_covers_chunk_set():
    a = atom("Routing")
    assert _cache_key(a, chunks_json("c1", "c2")).name != _cache_key(a, chunks_json("c1")).name


# ----------------------------- annotation -----------------------------

def test_atom_meta_fills_node_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("bachelor_s_project_i_lms.graph.atom_meta.CACHE_DIR", tmp_path)
    a = atom("Routing")
    atom_meta(a, generate=lambda m, s: (meta_json(["c1"]), {}), fetch=fake_fetch("c1", "c2"))
    assert a["learning_objective"].startswith("The learner can")
    assert a["estimated_time_min"] == 5
    assert a["difficulty_weight"] == 3
    assert a["type"] == "concept"
    assert len(a["common_misconceptions"]) == 2
    assert a["source_chunk_ids"] == ["c1"]


# ----------------------------- entry point -----------------------------

def test_run_annotates_and_checkpoints(tmp_path, monkeypatch):
    tree = [{
        "temp_id": "id_root", "title": "Root", "summary": "", "depth": 0,
        "root": "Root", "parent_temp_id": None, "is_atomic": False,
        "needs_review": False,
        "children": [atom("A"), atom("B")],
    }]
    tree_path = tmp_path / "tree.json"
    tree_path.write_text(json.dumps(tree))

    def fake_generate(messages, system):
        # both atoms retrieve the same chunks; metadata cites c1
        return meta_json(["c1"]), {}

    monkeypatch.setattr("bachelor_s_project_i_lms.graph.atom_meta.CACHE_DIR", tmp_path)
    atoms_path = tmp_path / "atoms.json"
    out_tree, n_flagged = run(generate=fake_generate, fetch=fake_fetch("c1"),
                               tree_path=tree_path, atoms_path=atoms_path)

    assert n_flagged == 0
    assert all(a.get("learning_objective") for a in out_tree[0]["children"])
    saved_tree = json.loads(tree_path.read_text())
    assert saved_tree[0]["children"][0]["source_chunk_ids"] == ["c1"]
    saved_atoms = json.loads(atoms_path.read_text())
    assert [a["title"] for a in saved_atoms] == ["A", "B"]
    assert "children" not in saved_atoms[0]


def test_run_skips_already_annotated(tmp_path, monkeypatch):
    pre = atom("A")
    pre["learning_objective"] = "The learner can do A."
    tree = [{
        "temp_id": "id_root", "title": "Root", "summary": "", "depth": 0,
        "root": "Root", "parent_temp_id": None, "is_atomic": False,
        "needs_review": False,
        "children": [pre, atom("B")],
    }]
    tree_path = tmp_path / "tree.json"
    tree_path.write_text(json.dumps(tree))

    calls = []

    def fake_generate(messages, system):
        calls.append(messages)
        return meta_json(["c1"]), {}

    monkeypatch.setattr("bachelor_s_project_i_lms.graph.atom_meta.CACHE_DIR", tmp_path)
    _, n_flagged = run(generate=fake_generate, fetch=fake_fetch("c1"),
                       tree_path=tree_path, atoms_path=tmp_path / "atoms.json")
    assert len(calls) == 1  # only atom B hit the LLM; A was already done
    assert n_flagged == 0


def test_atoms_roundtrip(tmp_path):
    atoms = [{"temp_id": "id_x", "title": "X", "learning_objective": "obj"}]
    save_atoms(atoms, tmp_path / "atoms.json")
    assert load_atoms(tmp_path / "atoms.json") == atoms

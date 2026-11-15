"""Phase 5: cross-branch prerequisite edges between the chapter islands.

The local pass (Phase 3) only ordered siblings, so the 8 chapter graphs are
disconnected. This phase stitches them together:

1. Embed every atom as ``title + learning objective`` (same bge model as the
   chunks; done in one batch, no LLM).
2. Candidates: for each atom, the top-k most similar atoms *from other roots*,
   plus — as a second, free generator — any cross-root pair sharing a
   ``source_chunk_id`` (two atoms grounded in the same textbook passage are the
   "same concept at two depths" case similarity can miss).
3. One LLM call per unique pair judges BOTH directions symmetrically; book
   position is evidence in the prompt, never a rule.
4. Code verifies: titles must exist, verdicts must be well-formed, same-root
   and self pairs are dropped defensively, and only edges with confidence >=
   0.7 are kept with ``origin = "cross_branch"``. A kept edge that points from
   a later book page to an earlier one is flagged ``backward`` for review.

Every LLM call is cached per pair so an interrupted run never pays twice.
Results land in ``checkpoints/cross_edges.json``; merging into ``edges.json``
happens only when the full pass completes cleanly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from bachelor_s_project_i_lms.graph.atom_meta import load_atoms
from bachelor_s_project_i_lms.graph.cross_branch_prompts import (
    CROSS_BRANCH_SYSTEM,
    build_cross_branch_messages,
)
from bachelor_s_project_i_lms.llm.client import call_llm
from bachelor_s_project_i_lms.llm.schemas import CrossBranchPairVerdict

CROSS_EDGES_PATH = Path(__file__).resolve().parent / "checkpoints" / "cross_edges.json"
CACHE_DIR = Path(__file__).resolve().parent / "checkpoints" / "llm_cache"  # shared with decompose

CONFIDENCE_THRESHOLD = 0.7
TOP_K = 8  # similar candidates per atom from other roots


# ----------------------------- book position -----------------------------

def _min_page(atom: dict) -> Optional[int]:
    """Earliest book page cited by the atom, or None. Chunk ids look like
    'chunk_p48_52', so the first number after 'p' is the chunk's start page."""
    import re

    pages = []
    for cid in atom.get("source_chunk_ids", []):
        m = re.search(r"p(\d+)", str(cid))
        if m:
            pages.append(int(m.group(1)))
    return min(pages) if pages else None


# ----------------------------- similarity -----------------------------

def _cosine(u: list[float], v: list[float]) -> float:
    """Dot product of two normalized vectors (bge embeddings are normalized,
    so this is cosine similarity). Pure Python: 139 atoms is a tiny workload."""
    return sum(a * b for a, b in zip(u, v))


def _norm_title(title: str) -> str:
    return " ".join(title.strip().lower().split())


def _proposals_for_atom(a: dict, atoms: list[dict], vectors: dict[str, list[float]], top_k: int) -> list[tuple[str, str]]:
    """The pairs one atom proposes: its top_k nearest atoms from other roots."""
    va = vectors.get(a["temp_id"])
    if va is None:
        return []
    scored = []
    for b in atoms:
        if b["root"] == a["root"] or b["temp_id"] == a["temp_id"]:
            continue
        vb = vectors.get(b["temp_id"])
        if vb is None:
            continue
        scored.append((_cosine(va, vb), b["temp_id"]))
    scored.sort(key=lambda t: (-t[0], t[1]))  # similarity desc, tie-break by id for determinism
    return [_ordered_pair(a["temp_id"], b_id) for _, b_id in scored[:top_k]]


def candidates(atoms: list[dict], vectors: dict[str, list[float]], top_k: int = TOP_K) -> list[tuple[str, str]]:
    """Unique cross-root candidate pairs (temp_id_a, temp_id_b), a < b.

    Two generators, unioned:
    - similarity: each atom proposes its top-k nearest atoms from other roots
      (the union is bidirectional — an atom can be proposed by others more
      often than top_k, which is fine: the cap is per atom, per spec);
    - chunk overlap: cross-root pairs sharing at least one source_chunk_id —
      grounded in the same passage, which similarity ranking can miss.
    """
    pairs: set[tuple[str, str]] = set()
    for a in atoms:
        pairs.update(_proposals_for_atom(a, atoms, vectors, top_k))

    by_chunk: dict[str, list[dict]] = {}
    for a in atoms:
        for cid in a.get("source_chunk_ids", []):
            by_chunk.setdefault(str(cid), []).append(a)
    for atoms_sharing in by_chunk.values():
        for i in range(len(atoms_sharing)):
            for j in range(i + 1, len(atoms_sharing)):
                x, y = atoms_sharing[i], atoms_sharing[j]
                if x["root"] != y["root"]:
                    pairs.add(_ordered_pair(x["temp_id"], y["temp_id"]))

    return sorted(pairs)


def _ordered_pair(x: str, y: str) -> tuple[str, str]:
    return (x, y) if x <= y else (y, x)


# ----------------------------- LLM + cache -----------------------------

def _cache_key(atom_a: dict, atom_b: dict) -> Path:
    """Cache key covers both titles AND objectives: a re-run of Phase 4 that
    changes an objective must not reuse a stale verdict."""
    parts = [
        _norm_title(atom_a["title"]),
        atom_a.get("learning_objective", ""),
        _norm_title(atom_b["title"]),
        atom_b.get("learning_objective", ""),
    ]
    h = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
    return CACHE_DIR / f"crossbranch_{h}.json"


def judge_pair(atom_a: dict, atom_b: dict, generate=None) -> CrossBranchPairVerdict:
    """One LLM call: both directions for the pair. Cached."""
    key = _cache_key(atom_a, atom_b)
    if key.exists():
        return CrossBranchPairVerdict.model_validate_json(key.read_text(encoding="utf-8"))

    messages = build_cross_branch_messages(atom_a, atom_b)
    out = call_llm(
        messages=messages,
        response_model=CrossBranchPairVerdict,
        system=CROSS_BRANCH_SYSTEM,
        generate=generate,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key.write_text(out.model_dump_json(indent=2), encoding="utf-8")
    return out


# ----------------------------- edge records -----------------------------

def _edge(from_atom: dict, to_atom: dict, confidence: float, rationale: str) -> dict:
    edge = {
        "from_temp_id": from_atom["temp_id"],
        "to_temp_id": to_atom["temp_id"],
        "confidence": confidence,
        "rationale": rationale,
        "origin": "cross_branch",
    }
    # A kept edge pointing from a later book page to an earlier one is rare and
    # surprising — exactly what a human should glance at in Phase 7.
    p_from, p_to = _min_page(from_atom), _min_page(to_atom)
    if p_from is not None and p_to is not None and p_from > p_to:
        edge["backward"] = True
    return edge


def verdicts_to_edges(atom_a: dict, atom_b: dict, verdict: CrossBranchPairVerdict) -> list[dict]:
    """Turn one pair verdict into 0, 1 or 2 edges, threshold applied in code."""
    edges = []
    if verdict.a_before_b.is_prerequisite and verdict.a_before_b.confidence >= CONFIDENCE_THRESHOLD:
        edges.append(_edge(atom_a, atom_b, verdict.a_before_b.confidence, verdict.a_before_b.rationale))
    if verdict.b_before_a.is_prerequisite and verdict.b_before_a.confidence >= CONFIDENCE_THRESHOLD:
        edges.append(_edge(atom_b, atom_a, verdict.b_before_a.confidence, verdict.b_before_a.rationale))
    return edges


# ----------------------------- reporting -----------------------------

def confidence_histogram(verdicts: dict[tuple[str, str], CrossBranchPairVerdict]) -> dict[str, int]:
    """Distribution of every directional confidence, so the 0.7 threshold can
    be checked against the real data before Phase 6 bakes the edges in."""
    buckets = {"0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
    for v in verdicts.values():
        for d in (v.a_before_b, v.b_before_a):
            c = d.confidence
            if c < 0.3:
                buckets["0.0-0.3"] += 1
            elif c < 0.5:
                buckets["0.3-0.5"] += 1
            elif c < 0.7:
                buckets["0.5-0.7"] += 1
            elif c < 0.9:
                buckets["0.7-0.9"] += 1
            else:
                buckets["0.9-1.0"] += 1
    return buckets


# ----------------------------- checkpoint -----------------------------

def save_cross_edges(edges: list[dict], path: Path = CROSS_EDGES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(edges, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cross_edges(path: Path = CROSS_EDGES_PATH) -> Optional[list[dict]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# ----------------------------- entry point -----------------------------

def run(
    generate=None,
    embed_fn=None,
    atoms_path=None,
    edges_path: Path = CROSS_EDGES_PATH,
    top_k: int = TOP_K,
) -> tuple[list[dict], dict[str, int]]:
    """Compute cross-branch edges for all atoms in the checkpointed catalog.

    Safe to interrupt: every LLM call is cached and edges are saved after every
    25 judged pairs. Returns (edges, confidence_histogram).
    """
    from bachelor_s_project_i_lms.graph.atom_meta import ATOMS_PATH

    atoms = load_atoms(atoms_path or ATOMS_PATH)
    if atoms is None:
        raise RuntimeError("no atoms.json found — run atom_meta first")

    by_id = {a["temp_id"]: a for a in atoms}

    if embed_fn is None:
        from bachelor_s_project_i_lms.scripts.vector_search import embed as embed_fn  # lazy: heavy import

    vectors = {a["temp_id"]: embed_fn(f"{a['title']}. {a.get('learning_objective', '')}") for a in atoms}

    pairs = candidates(atoms, vectors, top_k=top_k)
    print(f"= {len(pairs)} unique candidate pairs (top_k={top_k} + chunk overlap)")

    edges: list[dict] = []
    verdicts: dict[tuple[str, str], CrossBranchPairVerdict] = {}
    for i, (a_id, b_id) in enumerate(pairs, start=1):
        verdict = judge_pair(by_id[a_id], by_id[b_id], generate=generate)
        verdicts[(a_id, b_id)] = verdict
        edges.extend(verdicts_to_edges(by_id[a_id], by_id[b_id], verdict))
        if i % 25 == 0 or i == len(pairs):
            save_cross_edges(edges, edges_path)
            print(f"= judged {i}/{len(pairs)} pairs, {len(edges)} edges kept so far")

    hist = confidence_histogram(verdicts)
    print("= confidence histogram:", json.dumps(hist))
    n_backward = sum(1 for e in edges if e.get("backward"))
    if n_backward:
        print(f"= NOTE: {n_backward} kept edge(s) point backward in book order — review in Phase 7")
    return edges, hist


if __name__ == "__main__":
    run()

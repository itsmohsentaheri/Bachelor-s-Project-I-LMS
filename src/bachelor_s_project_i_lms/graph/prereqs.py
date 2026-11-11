"""Phase 3: local prerequisite edges among siblings, expanded to leaf-to-leaf edges.

For every non-leaf node we make one ``local_prereq`` LLM call asking for prerequisite
edges between its children. An edge between two *atomic* children is kept as-is.
When a non-atomic child A is a prerequisite of a non-atomic child B, the edge is
rewritten in code (never by the LLM) between leaves: the "output" leaves of A
(sinks of A's internal DAG — nothing in A depends on them) connect to the "input"
leaves of B (sources of B's internal DAG — no internal prerequisite feeds them).

All state lives next to the decomposition checkpoints: raw LLM outputs are cached
per (parent, children) so an interrupted run never pays twice, and the final
leaf-level edges are written to ``checkpoints/edges.json`` with temporary IDs
(stable node_ids are only created during freeze).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from bachelor_s_project_i_lms.graph.decompose import CHECKPOINT_PATH, _norm, load_checkpoint
from bachelor_s_project_i_lms.graph.prereq_prompts import LOCAL_PREREQ_SYSTEM, build_local_prereq_messages
from bachelor_s_project_i_lms.llm.client import call_llm
from bachelor_s_project_i_lms.llm.schemas import LocalPrereqOutput

EDGES_PATH = Path(__file__).resolve().parent / "checkpoints" / "edges.json"
CACHE_DIR = Path(__file__).resolve().parent / "checkpoints" / "llm_cache"  # shared with decompose


# ----------------------------- LLM + cache -----------------------------

def _cache_key(node: dict) -> Path:
    """Cache key covers the parent title AND its children: a re-decomposed parent
    (different children) must not reuse a stale verdict."""
    child_titles = sorted(_norm(c["title"]) for c in node["children"])
    h = hashlib.sha256("\x1f".join([_norm(node["title"]), str(node["depth"]), *child_titles]).encode()).hexdigest()
    return CACHE_DIR / f"prereq_{h}.json"


def local_prereqs(node: dict, generate=None) -> LocalPrereqOutput:
    """One LLM call: prerequisite edges among the children of `node`. Cached."""
    key = _cache_key(node)
    if key.exists():
        return LocalPrereqOutput.model_validate_json(key.read_text(encoding="utf-8"))

    messages = build_local_prereq_messages(node["title"], node["children"])
    out = call_llm(messages=messages, response_model=LocalPrereqOutput, system=LOCAL_PREREQ_SYSTEM, generate=generate)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key.write_text(out.model_dump_json(indent=2), encoding="utf-8")
    return out


# ----------------------------- leaf helpers -----------------------------

def _subtree_leaves(node: dict) -> list[dict]:
    if node["is_atomic"] or not node["children"]:
        return [node]
    out: list[dict] = []
    for c in node["children"]:
        out.extend(_subtree_leaves(c))
    return out


def _sinks(node: dict, edges: list[dict]) -> list[dict]:
    """Leaves of node's subtree that nothing (within the subtree) depends on:
    the terminal outcomes of learning this topic."""
    has_outgoing = {e["from_temp_id"] for e in edges}
    return [l for l in _subtree_leaves(node) if l["temp_id"] not in has_outgoing]


def _sources(node: dict, edges: list[dict]) -> list[dict]:
    """Leaves of node's subtree with no internal prerequisite: entry points."""
    has_incoming = {e["to_temp_id"] for e in edges}
    return [l for l in _subtree_leaves(node) if l["temp_id"] not in has_incoming]


# ----------------------------- edge records -----------------------------

def _dedupe(edges: list[dict]) -> list[dict]:
    """Same (from, to) pair may arise from several sibling edges: keep the best."""
    best: dict[tuple[str, str], dict] = {}
    for e in edges:
        k = (e["from_temp_id"], e["to_temp_id"])
        if k not in best or e["confidence"] > best[k]["confidence"]:
            best[k] = e
    return sorted(best.values(), key=lambda e: (e["from_temp_id"], e["to_temp_id"]))


# ----------------------------- recursion -----------------------------

def leaf_edges(node: dict, generate=None) -> list[dict]:
    """All leaf-level prerequisite edges inside `node`'s subtree.

    Post-order: each child's internal leaf edges are computed first, so the
    sibling-edge transformation can find sinks/sources inside their subtrees.
    """
    if not node["children"]:
        return []

    per_child = {c["temp_id"]: leaf_edges(c, generate=generate) for c in node["children"]}
    edges = [e for es in per_child.values() for e in es]

    by_norm = {_norm(c["title"]): c for c in node["children"]}
    for e in local_prereqs(node, generate=generate).edges:
        a = by_norm.get(_norm(e.prerequisite))
        b = by_norm.get(_norm(e.dependent))
        if a is None or b is None or a is b:
            continue  # model referenced something that is not a child (or itself) — drop
        a_edges = per_child[a["temp_id"]]
        b_edges = per_child[b["temp_id"]]
        transformed = not (a["is_atomic"] and b["is_atomic"])
        for src in _sinks(a, a_edges):
            for dst in _sources(b, b_edges):
                rationale = e.rationale
                if transformed:
                    rationale += f" [expanded from sibling edge '{a['title']}' -> '{b['title']}']"
                edges.append(
                    {
                        "from_temp_id": src["temp_id"],
                        "to_temp_id": dst["temp_id"],
                        "confidence": e.confidence,
                        "rationale": rationale,
                        "origin": "local",
                    }
                )
    return _dedupe(edges)


# ----------------------------- checkpoint -----------------------------

def save_edges(edges: list[dict], path: Path = EDGES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(edges, ensure_ascii=False, indent=2), encoding="utf-8")


def load_edges(path: Path = EDGES_PATH) -> Optional[list[dict]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# ----------------------------- entry point -----------------------------

def run(generate=None, tree_path: Path = CHECKPOINT_PATH, edges_path: Path = EDGES_PATH) -> list[dict]:
    """Compute leaf-level local edges for the whole checkpointed tree.

    Safe to interrupt: every LLM call is cached, and edges are saved after each
    root so a crash loses at most one chapter.
    """
    tree = load_checkpoint(tree_path)
    if tree is None:
        raise RuntimeError("no tree checkpoint found — run decompose first")

    all_edges: list[dict] = []
    for root in tree:
        all_edges.extend(leaf_edges(root, generate=generate))
        save_edges(all_edges, edges_path)
        print(f"= {root['title']}: {len(all_edges)} leaf edges so far")
    return all_edges


if __name__ == "__main__":
    run()

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

from bachelor_s_project_i_lms.db.session import SessionLocal
from bachelor_s_project_i_lms.db.models import BookChunk
from bachelor_s_project_i_lms.graph.decompose_prompts import DECOMPOSE_SYSTEM, build_decompose_messages
from bachelor_s_project_i_lms.llm.client import call_llm
from bachelor_s_project_i_lms.llm.schemas import DecomposeOutput


MAX_DEPTH = 4
TOP_K_CHUNKS = 8
COURSE_FILE = Path(__file__).resolve().parent.parent / "courses" / "computer_networks.md"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "tree.json"
CACHE_DIR = Path(__file__).resolve().parent / "checkpoints" / "llm_cache"


def _norm(title: str) -> str:
    """Normalize a title for cache keys and dedup checks."""
    s = unicodedata.normalize("NFKD", title).lower().strip()
    return re.sub(r"\s+", " ", s)


def _tid(*parts: str) -> str:
    """Short deterministic temp id (NOT the final node_id; that comes from freeze)."""
    return "t_" + hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:10]


# ----------------------------- step 1: roots -----------------------------

def load_roots(path: Path = COURSE_FILE) -> list[str]:
    """Parse first-level chapter titles from the course markdown (one root each)."""
    roots: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*\d+\s*[::]\s*(.+)$", line)
        if m:
            roots.append(m.group(1).strip())
    if not (6 <= len(roots) <= 10):
        raise ValueError(f"expected 6-10 root chapters, got {len(roots)} — fix the course file first")
    return roots


# ----------------------------- chunk retrieval -----------------------------

def fetch_chunks(query: str, k: int = TOP_K_CHUNKS) -> list[dict]:
    """Vector search book_chunks for a topic title. Returns [{chunk_id, page, text}]."""
    # local import: keeps sentence_transformers (a heavy ML dep) out of module import time
    from bachelor_s_project_i_lms.scripts.vector_search import embed

    qvec = embed(query)
    db = SessionLocal()
    try:
        rows = (
            db.query(BookChunk)
            .order_by(BookChunk.embedding.cosine_distance(qvec))
            .limit(k)
            .all()
        )
        return [{"chunk_id": c.chunk_id, "page": c.page_number, "text": c.content} for c in rows]
    finally:
        db.close()


# ----------------------------- LLM + cache -----------------------------

def _cache_key(title: str, depth: int) -> Path:
    h = hashlib.sha256(f"{_norm(title)}\x1f{depth}".encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def decompose_topic(title: str, summary_hint: str, depth: int, generate=None) -> DecomposeOutput:
    """Split one topic into 3-7 subtopics. Cached: identical (title, depth) never re-calls the LLM."""
    key = _cache_key(title, depth)
    if key.exists():
        return DecomposeOutput.model_validate_json(key.read_text(encoding="utf-8"))

    chunks = fetch_chunks(title)
    messages = build_decompose_messages(title=title, summary_hint=summary_hint, chunks=chunks, depth=depth)
    out = call_llm(messages=messages, response_model=DecomposeOutput, system=DECOMPOSE_SYSTEM, generate=generate)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key.write_text(out.model_dump_json(indent=2), encoding="utf-8")
    return out


# ----------------------------- recursion -----------------------------

def _new_node(title: str, summary: str, parent_tid: Optional[str], depth: int, root: str) -> dict:
    return {
        "temp_id": _tid(root, title),
        "title": title,
        "summary": summary,
        "depth": depth,
        "root": root,               # syllabus chapter this subtree belongs to (coverage check later)
        "parent_temp_id": parent_tid,
        "is_atomic": False,
        "needs_review": False,
        "children": [],
    }


def expand(node: dict, generate=None) -> dict:
    """Recursively expand `node` in place until leaves are atomic or MAX_DEPTH is hit."""
    title, depth = node["title"], node["depth"]
    print(f"{'  ' * depth}- {title} (depth {depth})")

    if depth >= MAX_DEPTH:
        # Depth cap reached: force it to be an atom, flag for human review (step 7).
        node["is_atomic"] = True
        node["needs_review"] = True
        return node

    out = decompose_topic(title, node.get("summary", ""), depth, generate=generate)

    seen = {_norm(title)}  # avoid child with same title as parent
    for st in out.subtopics:
        t = _norm(st.title)
        if t in seen:
            continue
        seen.add(t)
        child = _new_node(st.title, st.summary, node["temp_id"], depth + 1, node["root"])
        if st.is_atomic:
            child["is_atomic"] = True
        else:
            expand(child, generate=generate)
        node["children"].append(child)

    # If every child ended up merged/duplicate (rare), treat this node itself as an atom.
    if not node["children"]:
        node["is_atomic"] = True
        node["needs_review"] = True
    return node


# ----------------------------- checkpoint -----------------------------

def save_checkpoint(tree: list[dict], path: Path = CHECKPOINT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(path: Path = CHECKPOINT_PATH) -> Optional[list[dict]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def iter_atoms(tree: list[dict]):
    """Yield every leaf/atom node in the forest, in document order."""
    for node in tree:
        if node["is_atomic"]:
            yield node
        else:
            yield from iter_atoms(node["children"])


# ----------------------------- entry point -----------------------------

def run(generate=None, resume: bool = True) -> list[dict]:
    """Build the full topic tree for all roots. Safe to interrupt: results are cached."""
    roots = load_roots(COURSE_FILE)
    tree: list[dict] = []
    done = {n["temp_id"] for n in (load_checkpoint() or [])} if resume else set()

    for r in roots:
        root_node = _new_node(r, summary="", parent_tid=None, depth=0, root=r)
        if root_node["temp_id"] in done:
            print(f"= {r} (already in checkpoint, skipping)")
            continue
        expand(root_node, generate=generate)
        tree.append(root_node)
        save_checkpoint(tree)  # save after every root so a crash loses at most one chapter
    return tree


if __name__ == "__main__":
    run()

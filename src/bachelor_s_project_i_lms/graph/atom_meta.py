"""Phase 4: metadata for every atomic node (learning objective, time, difficulty,
misconceptions, source chunks).

For each atom we retrieve the relevant textbook chunks (vector search), make ONE
``call_llm(..., AtomMetadata)`` and attach the result to the node. The spec is strict
about provenance: any ``chunk_id`` the model returns that is not among the retrieved
(= database) chunks is rejected in code, never trusted from the model. If nothing the
model cites survives that check, the atom keeps no metadata and is flagged
``needs_review`` for the human pass (Phase 7) instead of silently inventing grounding.

Raw LLM outputs are cached per (atom, chunk set) so an interrupted run never pays
twice, and the enriched tree is checkpointed after every root.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from bachelor_s_project_i_lms.graph.decompose import (
    CHECKPOINT_PATH,
    _norm,
    fetch_chunks,
    iter_atoms,
    load_checkpoint,
    save_checkpoint,
)
from bachelor_s_project_i_lms.graph.atom_meta_prompts import ATOM_META_SYSTEM, build_atom_meta_messages
from bachelor_s_project_i_lms.llm.client import call_llm
from bachelor_s_project_i_lms.llm.schemas import AtomMetadata

ATOMS_PATH = Path(__file__).resolve().parent / "ai-engineering" / "checkpoints" / "atoms.json"
CACHE_DIR = Path(__file__).resolve().parent / "ai-engineering" / "checkpoints" / "llm_cache"  # shared with decompose


# ----------------------------- LLM + cache -----------------------------

def _cache_key(atom: dict, chunks: list[dict]) -> Path:
    """Cache key covers the atom title AND the retrieved chunk set: re-embedding or a
    changed retrieval must not reuse a stale metadata verdict."""
    chunk_ids = sorted(str(c["chunk_id"]) for c in chunks)
    h = hashlib.sha256("\x1f".join([_norm(atom["title"]), *chunk_ids]).encode()).hexdigest()
    return CACHE_DIR / f"atommeta_{h}.json"


def atom_meta(atom: dict, generate=None, fetch=None) -> Optional[AtomMetadata]:
    """Metadata for one atom, cached. Returns None (and flags the atom) when the model
    cites no valid database chunk: metadata without provenance is worthless."""
    chunks = (fetch or fetch_chunks)(atom["title"])
    key = _cache_key(atom, chunks)
    if key.exists():
        out = AtomMetadata.model_validate_json(key.read_text(encoding="utf-8"))
    else:
        messages = build_atom_meta_messages(atom, chunks)
        out = call_llm(messages=messages, response_model=AtomMetadata, system=ATOM_META_SYSTEM, generate=generate)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key.write_text(out.model_dump_json(indent=2), encoding="utf-8")

    # Provenance check in code: the model may only cite chunks we actually retrieved.
    valid_ids = {str(c["chunk_id"]) for c in chunks}
    out.source_chunk_ids = [cid for cid in out.source_chunk_ids if cid in valid_ids]
    if not out.source_chunk_ids:
        atom["needs_review"] = True
        return None

    atom["learning_objective"] = out.learning_objective
    atom["estimated_time_min"] = out.estimated_time_min
    atom["difficulty_weight"] = out.difficulty_weight
    atom["type"] = out.type
    atom["common_misconceptions"] = [m.model_dump() for m in out.common_misconceptions]
    atom["source_chunk_ids"] = out.source_chunk_ids
    return out


# ----------------------------- checkpoint -----------------------------

def save_atoms(atoms: list[dict], path: Path = ATOMS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(atoms, ensure_ascii=False, indent=2), encoding="utf-8")


def load_atoms(path: Path = ATOMS_PATH) -> Optional[list[dict]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# ----------------------------- entry point -----------------------------

def run(generate=None, fetch=None, tree_path: Path = CHECKPOINT_PATH, atoms_path: Path = ATOMS_PATH) -> tuple[list[dict], int]:
    """Fill metadata for every atom in the checkpointed tree.

    Already-annotated atoms are skipped, results are checkpointed after every root, and
    every LLM call is cached — safe to interrupt and resume.
    Returns (tree, n_flagged) where n_flagged counts atoms left for human review.
    """
    tree = load_checkpoint(tree_path)
    if tree is None:
        raise RuntimeError("no tree checkpoint found — run decompose first")

    n_flagged = 0
    done_atoms: list[dict] = []
    for root in tree:
        root_atoms = list(iter_atoms([root]))
        for atom in root_atoms:
            if atom.get("learning_objective"):
                continue  # already annotated (previous interrupted run)
            if atom_meta(atom, generate=generate, fetch=fetch) is None:
                n_flagged += 1
        save_checkpoint(tree, tree_path)
        done_atoms.extend({k: a[k] for k in a if k != "children"} for a in root_atoms)
        save_atoms(done_atoms, atoms_path)
        print(f"= {root['title']}: {sum(1 for a in root_atoms if a.get('learning_objective'))}/{len(root_atoms)} atoms annotated")
    return tree, n_flagged


if __name__ == "__main__":
    run()

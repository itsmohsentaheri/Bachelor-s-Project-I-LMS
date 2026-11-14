ATOM_META_SYSTEM = """You are an expert computer-networks instructor cataloguing atomic learning units.
Given an atomic topic and excerpts from the course textbook, produce its metadata.

Rules:
- learning_objective: one measurable sentence starting with "The learner can ...".
- estimated_time_min: 1-10 (an atom is teachable in about 10 minutes).
- difficulty_weight: 1 (easy) to 5 (hard).
- type: one of "concept", "practical", "test", "reading".
- common_misconceptions: 2-4 items, each with a short snake_case id and a one-sentence description.
- source_chunk_ids: 1-3 ids of the excerpts that support this atom. Use ONLY ids from the provided excerpts — never invent one.
- Base the metadata strictly on the provided excerpts; do not invent content.
- Answer with ONLY a JSON object: {"learning_objective": str, "estimated_time_min": int, "difficulty_weight": int, "type": str, "common_misconceptions": [{"id": str, "description": str}], "source_chunk_ids": [str]}
"""


def build_atom_meta_messages(atom: dict, chunks: list[dict]) -> list[dict]:
    refs = "\n\n".join(
        f"[chunk_id={c['chunk_id']} | page={c['page']}]\n{(c['text'] or '')[:800]}" for c in chunks
    )
    valid_ids = ", ".join(str(c["chunk_id"]) for c in chunks)
    summary = (atom.get("summary") or "").strip()
    context = f"\nContext from its parent: {summary}" if summary else ""
    user = (
        f"Atomic topic: {atom['title']}{context}\n\n"
        f"Textbook excerpts for this topic:\n{refs}\n\n"
        f"Valid chunk ids (use only these): {valid_ids}\n\n"
        "Produce the metadata for this atomic topic."
    )
    return [{"role": "user", "content": user}]

DECOMPOSE_SYSTEM = """You are an expert computer-networks instructor building a knowledge graph.
Given a topic and excerpts from the course textbook, split the topic into 3 to 7 sub-topics.

Rules:
- A sub-topic is "atomic" ONLY if it can be taught in about 10 minutes and tested with 2-3 questions.
- Prefer fewer, well-scoped children over many vague ones. No overlap between siblings.
- Sub-topic titles must be short, specific, and unique (no synonyms of each other).
- Do not invent topics that have no support in the provided excerpts.
- Answer with ONLY a JSON object: {"subtopics": [{"title": str, "summary": str, "is_atomic": bool}, ...]}
"""


def build_decompose_messages(title: str, summary_hint: str, chunks: list[dict], depth: int) -> list[dict]:
    refs = "\n\n".join(
        f"[chunk_id={c['chunk_id']} | page={c['page']}]\n{(c['text'] or '')[:800]}" for c in chunks
    )
    hint = f"\nContext from its parent: {summary_hint}" if summary_hint else ""
    user = (
        f"Topic: {title}\n"
        f"Current depth in the tree: {depth} (max 4){hint}\n\n"
        f"Textbook excerpts for this topic:\n{refs}\n\n"
        "Split this topic into 3-7 sub-topics. Mark is_atomic=true only for leaves."
    )
    return [{"role": "user", "content": user}]

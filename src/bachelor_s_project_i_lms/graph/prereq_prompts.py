LOCAL_PREREQ_SYSTEM = """You are an expert computer-networks instructor building a prerequisite graph.
Given a parent topic and its immediate children, decide which children are prerequisites for which other children.

Rules:
- An edge X -> Y means: to understand Y, X should be learned first.
- Only propose edges between the listed children, using their exact titles.
- Only propose an edge for a real, specific dependency — not thematic similarity or same-chapter proximity.
- A child can depend on several others; propose each dependency as its own edge.
- If the children are independent, return an empty edge list.
- Answer with ONLY a JSON object: {"edges": [{"prerequisite": str, "dependent": str, "confidence": float, "rationale": str}]}
"""


def build_local_prereq_messages(parent_title: str, children: list[dict]) -> list[dict]:
    lines = []
    for c in children:
        kind = "atomic leaf" if c["is_atomic"] else "compound (has subtopics)"
        summary = (c.get("summary") or "").strip()
        lines.append(f"- {c['title']} ({kind}): {summary}")
    user = (
        f"Parent topic: {parent_title}\n\n"
        f"Children:\n{''.join(line + chr(10) for line in lines)}\n"
        "List the prerequisite edges among these children (prerequisite -> dependent). "
        "Use exact child titles; return an empty list if none."
    )
    return [{"role": "user", "content": user}]

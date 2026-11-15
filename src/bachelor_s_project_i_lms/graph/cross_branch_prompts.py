"""Phase 5 prompt: cross-branch prerequisite verdict for one candidate pair.

Both atoms are presented symmetrically (same fields, same order, no pre-drawn
arrow and no "earlier/later" framing) so the model judges direction from the
content. Book position is included as plain evidence — knowing one atom is a
chapter-1 intro and the other a chapter-5 topic is legitimate signal — but the
prompt never suggests that page order implies dependency.
"""

CROSS_BRANCH_SYSTEM = """You are an expert computer-networks instructor building a prerequisite graph.
Given two atomic learning units from DIFFERENT chapters of the course, decide for each direction
whether mastering one is a prerequisite for understanding the other.

Rules:
- A prerequisite means: without first mastering X, a learner cannot meaningfully understand Y.
  Merely being related, similar, or taught in the same area is NOT a prerequisite.
- Judge each direction independently. Usually at most one direction is true; often neither is.
- Use the learning objectives (what the learner must be able to do), not just the titles.
- The chapter and page numbers are evidence of depth and placement, NOT of dependency order:
  a later chapter can still be independent, and an occasional true dependency can point backward.
- confidence: 0.0-1.0, how sure you are of that direction's verdict.
- rationale: one sentence per direction.
- Answer with ONLY a JSON object:
  {"a_before_b": {"is_prerequisite": bool, "confidence": 0.0-1.0, "rationale": str},
   "b_before_a": {"is_prerequisite": bool, "confidence": 0.0-1.0, "rationale": str}}
"""


def _atom_block(atom: dict) -> str:
    """Symmetric presentation: identical fields in identical order for both atoms."""
    pages = _page_range(atom)
    place = f"chapter: {atom['root']}"
    if pages:
        place += f", book pages: {pages}"
    return (
        f"Title: {atom['title']}\n"
        f"Placement: {place}\n"
        f"Learning objective: {atom.get('learning_objective', '')}"
    )


def _page_range(atom: dict) -> str:
    """'p48-55' from the atom's source chunks; empty when unparseable."""
    import re

    pages = []
    for cid in atom.get("source_chunk_ids", []):
        m = re.search(r"p(\d+)", str(cid))
        if m:
            pages.append(int(m.group(1)))
    if not pages:
        return ""
    return f"p{min(pages)}-{max(pages)}"


def build_cross_branch_messages(atom_a: dict, atom_b: dict) -> list[dict]:
    user = (
        f"Atom A\n{_atom_block(atom_a)}\n\n"
        f"Atom B\n{_atom_block(atom_b)}\n\n"
        "Q1: Is A a prerequisite for B? (a_before_b)\n"
        "Q2: Is B a prerequisite for A? (b_before_a)"
    )
    return [{"role": "user", "content": user}]

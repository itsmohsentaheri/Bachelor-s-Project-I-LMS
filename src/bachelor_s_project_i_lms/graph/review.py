"""Phase 7: export the validated graph for human review, then apply the edits.

The human pass needs the graph in a readable form and a place to record decisions:

1. ``nodes.csv`` — one row per atom with its metadata, degrees and prerequisite /
   dependent titles, so each atom can be checked for "testable in ~10 minutes".
2. ``graph.mmd`` — a Mermaid flowchart grouped by chapter, with confidence and
   origin on every edge and a visual flag on ``needs_review`` / isolated atoms.
3. ``review_edits.json`` — the editable file. It starts as a template listing the
   ``needs_review`` atoms; a reviewer fills in ``node_overrides``, ``manual_edges``
   (recorded with ``origin = "manual"``) and ``removed_edges``.

Applying the edits writes ``reviewed_atoms.json`` + ``reviewed_edges.json`` — the
inputs Phase 8 freezes. Edits are validated in code (unknown ids, self-loops,
duplicate edges, and any cycle introduced by a manual edge are rejected), because
a cycle would break the Phase 1 gate.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from bachelor_s_project_i_lms.graph.atom_meta import load_atoms
from bachelor_s_project_i_lms.graph.validate import _sccs

CHECKPOINTS = Path(__file__).resolve().parent / "ai-engineering" / "checkpoints"
NODES_CSV = CHECKPOINTS / "nodes.csv"
GRAPH_MMD = CHECKPOINTS / "graph.mmd"
EDITS_PATH = CHECKPOINTS / "review_edits.json"
REVIEWED_ATOMS_PATH = CHECKPOINTS / "reviewed_atoms.json"
REVIEWED_EDGES_PATH = CHECKPOINTS / "reviewed_edges.json"
REVIEW_REPORT_PATH = CHECKPOINTS / "review_report.json"

NODE_OVERRIDE_FIELDS = {
    "title",
    "summary",
    "learning_objective",
    "estimated_time_min",
    "difficulty_weight",
    "type",
    "common_misconceptions",
    "needs_review",
    "source_chunk_ids",
}

CSV_FIELDS = [
    "temp_id",
    "title",
    "root",
    "parent_temp_id",
    "depth",
    "type",
    "difficulty_weight",
    "estimated_time_min",
    "needs_review",
    "in_degree",
    "out_degree",
    "isolated",
    "prerequisites",
    "dependents",
    "learning_objective",
    "source_chunk_ids",
]


# ----------------------------- export -----------------------------

def _degrees(atoms: list[dict], edges: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    incoming: dict[str, int] = defaultdict(int)
    outgoing: dict[str, int] = defaultdict(int)
    for e in edges:
        outgoing[e["from_temp_id"]] += 1
        incoming[e["to_temp_id"]] += 1
    return incoming, outgoing


def export_nodes_csv(atoms: list[dict], edges: list[dict], path: Path = NODES_CSV) -> Path:
    """One row per atom, with prerequisite/dependent titles for the review pass."""
    by_id = {a["temp_id"]: a for a in atoms}
    incoming, outgoing = _degrees(atoms, edges)
    prereqs: dict[str, list[str]] = defaultdict(list)
    dependents: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        prereqs[e["to_temp_id"]].append(by_id[e["from_temp_id"]]["title"])
        dependents[e["from_temp_id"]].append(by_id[e["to_temp_id"]]["title"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for a in atoms:  # document order: chapter order, then depth
            writer.writerow({
                "temp_id": a["temp_id"],
                "title": a["title"],
                "root": a["root"],
                "parent_temp_id": a.get("parent_temp_id") or "",
                "depth": a.get("depth", ""),
                "type": a.get("type", ""),
                "difficulty_weight": a.get("difficulty_weight", ""),
                "estimated_time_min": a.get("estimated_time_min", ""),
                "needs_review": bool(a.get("needs_review")),
                "in_degree": incoming[a["temp_id"]],
                "out_degree": outgoing[a["temp_id"]],
                "isolated": incoming[a["temp_id"]] == 0 and outgoing[a["temp_id"]] == 0,
                "prerequisites": " | ".join(sorted(prereqs[a["temp_id"]])),
                "dependents": " | ".join(sorted(dependents[a["temp_id"]])),
                "learning_objective": a.get("learning_objective", ""),
                "source_chunk_ids": " | ".join(a.get("source_chunk_ids", [])),
            })
    return path


def _label(text: str) -> str:
    """Mermaid-safe label text (quoted labels still break on brackets/quotes)."""
    return (text.replace('"', "'").replace("[", "(").replace("]", ")")
            .replace("\n", " ").strip())


def export_mermaid(atoms: list[dict], edges: list[dict], path: Path = GRAPH_MMD) -> Path:
    """Flowchart grouped by chapter; edges carry confidence + origin."""
    by_id = {a["temp_id"]: a for a in atoms}
    incoming, outgoing = _degrees(atoms, edges)
    roots: dict[str, list[dict]] = defaultdict(list)
    for a in atoms:
        roots[a["root"]].append(a)

    lines = ["flowchart TD"]
    for i, (root, members) in enumerate(roots.items()):
        lines.append(f'  subgraph sg{i}["{_label(root)}"]')
        for a in sorted(members, key=lambda x: (x["depth"], x["title"])):
            bits = [a["title"]]
            meta = []
            if a.get("estimated_time_min"):
                meta.append(f"{a['estimated_time_min']} min")
            if a.get("difficulty_weight"):
                meta.append(f"d{a['difficulty_weight']}")
            if a.get("type"):
                meta.append(a["type"])
            if meta:
                bits.append(" · ".join(meta))
            if a.get("needs_review"):
                bits.append("needs review")
            lines.append(f'    {a["temp_id"]}["{_label("<br/>".join(bits))}"]')
        lines.append("  end")

    for e in sorted(edges, key=lambda x: (x["from_temp_id"], x["to_temp_id"])):
        lines.append(
            f'  {e["from_temp_id"]} -->|"{e["confidence"]:.2f} {e.get("origin", "local")}"| '
            f'{e["to_temp_id"]}'
        )

    flagged = [a["temp_id"] for a in atoms if a.get("needs_review")]
    isolated = [a["temp_id"] for a in atoms
                if incoming[a["temp_id"]] == 0 and outgoing[a["temp_id"]] == 0]
    lines.append("  classDef needsReview fill:#ffe0e0,stroke:#c00,stroke-width:2px")
    lines.append("  classDef isolated fill:#fff4cc,stroke:#c90")
    if flagged:
        lines.append(f"  class {','.join(flagged)} needsReview")
    if isolated:
        lines.append(f"  class {','.join(isolated)} isolated")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_template(atoms: list[dict], path: Path = EDITS_PATH) -> Path:
    """Create the editable review file; never overwrite an existing one."""
    if path.exists():
        return path
    template = {
        "_instructions": (
            "Edit this file, then re-run review.apply_edits(). "
            "node_overrides: patch fields of one atom (set needs_review=false when fixed). "
            "manual_edges: add {from_temp_id, to_temp_id, rationale, confidence?}; "
            "origin is forced to 'manual'. removed_edges: list {from_temp_id, to_temp_id}."
        ),
        "node_overrides": {
            a["temp_id"]: {"needs_review": True}
            for a in atoms if a.get("needs_review")
        },
        "manual_edges": [],
        "removed_edges": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


# ----------------------------- apply edits -----------------------------

def apply_edits(atoms: list[dict], edges: list[dict], edits: dict) -> tuple[list[dict], list[dict], dict]:
    """Apply node overrides, removals and manual edges. Returns (atoms, edges, report).

    Never raises: structural problems land in ``report["errors"]`` and any cycle
    in the edited graph in ``report["cycles"]`` so the caller can refuse to write.
    """
    atoms = [dict(a) for a in atoms]
    edges = [dict(e) for e in edges]
    by_id = {a["temp_id"]: a for a in atoms}
    report: dict = {
        "node_overrides_applied": [],
        "manual_edges_added": [],
        "edges_removed": [],
        "errors": [],
        "cycles": [],
    }

    for tid, patch in (edits.get("node_overrides") or {}).items():
        if tid not in by_id:
            report["errors"].append(f"node_overrides: unknown temp_id {tid!r}")
            continue
        unknown = sorted(set(patch) - NODE_OVERRIDE_FIELDS)
        if unknown:
            report["errors"].append(f"node_overrides[{tid}]: unknown fields {unknown}")
            continue
        by_id[tid].update(patch)
        report["node_overrides_applied"].append({"temp_id": tid, "fields": sorted(patch)})

    for item in edits.get("removed_edges") or []:
        pair = (item.get("from_temp_id"), item.get("to_temp_id"))
        before = len(edges)
        edges = [e for e in edges if (e["from_temp_id"], e["to_temp_id"]) != pair]
        if len(edges) == before:
            report["errors"].append(f"removed_edges: edge {pair} not found")
        else:
            report["edges_removed"].append({"from_temp_id": pair[0], "to_temp_id": pair[1]})

    existing = {(e["from_temp_id"], e["to_temp_id"]) for e in edges}
    for item in edits.get("manual_edges") or []:
        u, v = item.get("from_temp_id"), item.get("to_temp_id")
        if u not in by_id or v not in by_id:
            report["errors"].append(f"manual_edges: unknown temp_id in {(u, v)}")
            continue
        if u == v:
            report["errors"].append(f"manual_edges: self-loop {u}")
            continue
        if (u, v) in existing:
            report["errors"].append(f"manual_edges: edge {(u, v)} already exists")
            continue
        edge = {
            "from_temp_id": u,
            "to_temp_id": v,
            "confidence": float(item.get("confidence", 1.0)),
            "rationale": item.get("rationale", "manual review"),
            "origin": "manual",
        }
        edges.append(edge)
        existing.add((u, v))
        report["manual_edges_added"].append(
            {"from": by_id[u]["title"], "to": by_id[v]["title"], "confidence": edge["confidence"]}
        )

    nodes = [a["temp_id"] for a in atoms]
    self_ids = {e["from_temp_id"] for e in edges if e["from_temp_id"] == e["to_temp_id"]}
    for comp in _sccs(nodes, edges):
        if len(comp) > 1 or comp[0] in self_ids:
            report["cycles"].append([by_id[n]["title"] for n in comp])

    return atoms, edges, report


# ----------------------------- entry point -----------------------------

def run(
    atoms_path: Optional[Path] = None,
    edges_path: Optional[Path] = None,
    nodes_csv: Path = NODES_CSV,
    mermaid_path: Path = GRAPH_MMD,
    edits_path: Path = EDITS_PATH,
    reviewed_atoms_path: Path = REVIEWED_ATOMS_PATH,
    reviewed_edges_path: Path = REVIEWED_EDGES_PATH,
    report_path: Path = REVIEW_REPORT_PATH,
) -> dict:
    """Export CSV + Mermaid + template, then apply ``review_edits.json``."""
    from bachelor_s_project_i_lms.graph.validate import FINAL_EDGES_PATH

    atoms = load_atoms(atoms_path) if atoms_path is not None else load_atoms()
    if atoms is None:
        raise RuntimeError("missing atoms checkpoint — run atom_meta first")
    edges_file = edges_path or FINAL_EDGES_PATH
    if not edges_file.exists():
        raise RuntimeError("missing final_edges.json — run validate first")
    edges = json.loads(edges_file.read_text(encoding="utf-8"))

    export_nodes_csv(atoms, edges, nodes_csv)
    export_mermaid(atoms, edges, mermaid_path)
    write_template(atoms, edits_path)

    edits = json.loads(edits_path.read_text(encoding="utf-8")) if edits_path.exists() else {}
    reviewed_atoms, reviewed_edges, report = apply_edits(atoms, edges, edits)
    report["n_atoms"] = len(reviewed_atoms)
    report["n_edges"] = len(reviewed_edges)
    report["n_flagged"] = sum(1 for a in reviewed_atoms if a.get("needs_review"))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"= nodes CSV: {nodes_csv} ({len(reviewed_atoms)} atoms)")
    print(f"= Mermaid:   {mermaid_path} ({len(reviewed_edges)} edges)")
    print(f"= edits:     {edits_path}")
    print(f"= applied:   {len(report['node_overrides_applied'])} overrides, "
          f"{len(report['manual_edges_added'])} manual edges, "
          f"{len(report['edges_removed'])} removals")
    print(f"= still flagged for review: {report['n_flagged']}")

    if report["errors"] or report["cycles"]:
        for err in report["errors"]:
            print(f"! {err}")
        for cycle in report["cycles"]:
            print(f"! cycle after edits: {' -> '.join(cycle)}")
        raise RuntimeError("review edits rejected — fix review_edits.json and re-run")

    reviewed_atoms_path.write_text(json.dumps(reviewed_atoms, ensure_ascii=False, indent=2), encoding="utf-8")
    reviewed_edges_path.write_text(json.dumps(reviewed_edges, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"= reviewed:  {reviewed_atoms_path.name}, {reviewed_edges_path.name}")
    return report


if __name__ == "__main__":
    run()

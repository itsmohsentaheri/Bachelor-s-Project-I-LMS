"""Phase 8: freeze the reviewed graph and publish it as ``graph_v1``.

The reviewed checkpoints are the single source of truth for what gets published:

1. Verify they are a faithful delta of the Phase 6 output — ``reviewed_atoms`` /
   ``reviewed_edges`` must equal ``apply_edits(atoms, final_edges, review_edits)``.
   This catches a stale review file (e.g. ``validate`` re-run after review).
2. Run the Phase 1 gate on the graph being frozen: required metadata, valid
   types/ranges, known edge endpoints, no self-loops/duplicates/cycles, every
   chapter owns at least one atom, and every ``source_ref`` chunk exists in the
   database.
3. Give every atom a permanent ``node_id`` = hash(normalized learning_objective +
   sorted source_ref). Temp ids are tree artifacts; this id survives rebuilds.
4. Insert ``graph_versions`` (draft), ``nodes`` and ``edges`` in ONE transaction,
   then set the version to ``frozen``. Any failure rolls everything back, so a
   half-published version can never exist.

The checkpoints on disk are never modified: they remain the audit trail.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Optional

from sqlalchemy import text

from bachelor_s_project_i_lms.graph import review
from bachelor_s_project_i_lms.graph.atom_meta import ATOMS_PATH, load_atoms
from bachelor_s_project_i_lms.graph.decompose import CHECKPOINT_PATH
from bachelor_s_project_i_lms.graph.validate import FINAL_EDGES_PATH, _sccs

CHECKPOINTS = Path(__file__).resolve().parent / "ai-engineering" / "checkpoints"
REVIEWED_ATOMS_PATH = CHECKPOINTS / "reviewed_atoms.json"
REVIEWED_EDGES_PATH = CHECKPOINTS / "reviewed_edges.json"
REPORT_PATH = CHECKPOINTS / "freeze_report.json"

GRAPH_VERSION = 1
NODE_TYPES = {"concept", "practical", "test", "reading"}
EDGE_ORIGINS = {"local", "cross_branch", "manual"}
NODE_ID_HEX = 24  # 96 bits: collision-safe for any realistic graph size


# ----------------------------- identity -----------------------------

def normalize_text(value: str) -> str:
    """NFKD, lowercase, whitespace-collapsed — the same atom must hash the same."""
    s = unicodedata.normalize("NFKD", value or "").lower().strip()
    return " ".join(s.split())


def node_id_for(atom: dict) -> str:
    """Stable id from normalized learning objective + sorted source chunks."""
    refs = ",".join(sorted(str(c) for c in atom.get("source_chunk_ids") or []))
    payload = f"{normalize_text(atom.get('learning_objective', ''))}\x1f{refs}"
    return "n_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:NODE_ID_HEX]


# ----------------------------- row building -----------------------------

def build_node_rows(atoms: list[dict], version: int = GRAPH_VERSION) -> tuple[list[dict], dict[str, str]]:
    """DB rows for ``nodes`` plus the temp_id -> node_id map used by edges."""
    rows: list[dict] = []
    id_map: dict[str, str] = {}
    for a in atoms:
        node_id = node_id_for(a)
        id_map[a["temp_id"]] = node_id
        rows.append({
            "node_id": node_id,
            "graph_version": version,
            "title": a["title"],
            "learning_objective": a["learning_objective"],
            "type": a["type"],
            "difficulty_weight": a["difficulty_weight"],
            "estimated_time_min": a["estimated_time_min"],
            "source_ref": list(a.get("source_chunk_ids") or []),
            "misconceptions": json.dumps(a.get("common_misconceptions") or [], ensure_ascii=False),
            "needs_review": bool(a.get("needs_review")),
            "parent_topic": a.get("root"),
        })
    return rows, id_map


def build_edge_rows(edges: list[dict], id_map: dict[str, str],
                    version: int = GRAPH_VERSION) -> list[dict]:
    return [{
        "graph_version": version,
        "from_id": id_map[e["from_temp_id"]],
        "to_id": id_map[e["to_temp_id"]],
        "confidence": float(e["confidence"]),
        "rationale": e.get("rationale", ""),
        "origin": e.get("origin", "local"),
    } for e in edges]


# ----------------------------- gate -----------------------------

def check_gate(atoms: list[dict], edges: list[dict],
               chunk_ids: Optional[set[str]] = None,
               tree_roots: Optional[list[str]] = None) -> dict:
    """Phase 1 gate on the graph about to be frozen. Returns {errors, warnings, stats}."""
    errors: list[str] = []
    warnings: list[str] = []
    by_id = {a["temp_id"]: a for a in atoms}

    for a in atoms:
        tid = a.get("temp_id", "<missing>")
        for field in ("title", "learning_objective", "type", "difficulty_weight",
                      "estimated_time_min", "root"):
            if a.get(field) in (None, ""):
                errors.append(f"{tid}: missing {field}")
        if a.get("type") not in NODE_TYPES:
            errors.append(f"{tid}: invalid type {a.get('type')!r}")
        difficulty = a.get("difficulty_weight")
        if not isinstance(difficulty, int) or not 1 <= difficulty <= 5:
            errors.append(f"{tid}: difficulty_weight must be 1..5, got {difficulty!r}")
        minutes = a.get("estimated_time_min")
        if not isinstance(minutes, int) or minutes <= 0:
            errors.append(f"{tid}: estimated_time_min must be > 0, got {minutes!r}")
        if not a.get("source_chunk_ids"):
            errors.append(f"{tid}: no source_ref (provenance required)")
        elif chunk_ids is not None:
            missing = sorted(set(a["source_chunk_ids"]) - chunk_ids)
            if missing:
                errors.append(f"{tid}: source_ref not in book_chunks: {missing}")

    node_ids = [node_id_for(a) for a in atoms]
    for node_id, count in Counter(node_ids).items():
        if count > 1:
            errors.append(f"duplicate node_id {node_id} ({count} atoms)")
    for temp_id, count in Counter(a.get("temp_id") for a in atoms).items():
        if count > 1:
            errors.append(f"duplicate temp_id {temp_id} ({count} atoms)")

    seen_pairs: set[tuple[str, str]] = set()
    for e in edges:
        u, v = e.get("from_temp_id"), e.get("to_temp_id")
        if u not in by_id or v not in by_id:
            errors.append(f"edge {u}->{v}: unknown endpoint")
            continue
        if u == v:
            errors.append(f"edge {u}->{v}: self-loop")
        if (u, v) in seen_pairs:
            errors.append(f"edge {u}->{v}: duplicate")
        seen_pairs.add((u, v))
        if e.get("origin") not in EDGE_ORIGINS:
            errors.append(f"edge {u}->{v}: invalid origin {e.get('origin')!r}")
        confidence = e.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(f"edge {u}->{v}: confidence must be 0..1, got {confidence!r}")

    nodes = list(by_id)
    self_ids = {e["from_temp_id"] for e in edges if e["from_temp_id"] == e["to_temp_id"]}
    for comp in _sccs(nodes, edges):
        if len(comp) > 1 or comp[0] in self_ids:
            errors.append("cycle: " + " -> ".join(by_id[n]["title"] for n in comp))

    roots = Counter(a.get("root") for a in atoms)
    for root, count in roots.items():
        if not root:
            errors.append("atom with empty root")
    if tree_roots is not None:
        missing_roots = sorted(set(tree_roots) - set(roots))
        if missing_roots:
            errors.append(f"syllabus headings without atoms: {missing_roots}")
    flagged = [a["temp_id"] for a in atoms if a.get("needs_review")]
    if flagged:
        warnings.append(f"{len(flagged)} atoms still needs_review: {flagged}")

    return {
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "n_atoms": len(atoms),
            "n_edges": len(edges),
            "n_roots": len(roots),
            "n_flagged": len(flagged),
            "edge_origins": dict(Counter(e.get("origin") for e in edges)),
        },
    }


# ----------------------------- staleness -----------------------------

def check_staleness(reviewed_atoms: list[dict], reviewed_edges: list[dict],
                    atoms: list[dict], final_edges: list[dict], edits: dict) -> list[str]:
    """Reviewed files must equal re-applying the edits to the Phase 6 output."""
    expected_atoms, expected_edges, _ = review.apply_edits(atoms, final_edges, edits)
    errors: list[str] = []

    if {a["temp_id"] for a in reviewed_atoms} != {a["temp_id"] for a in expected_atoms}:
        errors.append("reviewed_atoms: temp_id set differs from atoms.json")
    else:
        by_id = {a["temp_id"]: a for a in expected_atoms}
        for a in reviewed_atoms:
            if a != by_id[a["temp_id"]]:
                errors.append(f"reviewed_atoms[{a['temp_id']}]: differs from atoms + review_edits")

    def pairs(items: list[dict]) -> set[tuple[str, str]]:
        return {(e["from_temp_id"], e["to_temp_id"]) for e in items}

    if pairs(reviewed_edges) != pairs(expected_edges):
        errors.append("reviewed_edges: edge set differs from final_edges + review_edits")
    else:
        by_pair = {(e["from_temp_id"], e["to_temp_id"]): e for e in expected_edges}
        for e in reviewed_edges:
            if e != by_pair[(e["from_temp_id"], e["to_temp_id"])]:
                errors.append(f"reviewed_edges[{(e['from_temp_id'], e['to_temp_id'])}]: differs")

    return errors


# ----------------------------- publish -----------------------------

INSERT_VERSION = text(
    "INSERT INTO graph_versions (version, status, notes) VALUES (:version, 'draft', :notes)"
)
INSERT_NODES = text("""
    INSERT INTO nodes (node_id, graph_version, title, learning_objective, type,
                       difficulty_weight, estimated_time_min, source_ref,
                       misconceptions, needs_review, parent_topic)
    VALUES (:node_id, :graph_version, :title, :learning_objective, :type,
            :difficulty_weight, :estimated_time_min, :source_ref,
            CAST(:misconceptions AS jsonb), :needs_review, :parent_topic)
""")
INSERT_EDGES = text("""
    INSERT INTO edges (graph_version, from_id, to_id, confidence, rationale, origin)
    VALUES (:graph_version, :from_id, :to_id, :confidence, :rationale, :origin)
""")
FREEZE_VERSION = text(
    "UPDATE graph_versions SET status = 'frozen', frozen_at = now() WHERE version = :version"
)


def publish(conn, atoms: list[dict], edges: list[dict], version: int = GRAPH_VERSION,
            notes: str = "") -> dict:
    """Insert and freeze one version. ``conn`` must be inside a transaction."""
    existing = conn.execute(
        text("SELECT status FROM graph_versions WHERE version = :version"),
        {"version": version},
    ).first()
    if existing is not None:
        raise RuntimeError(f"graph_version {version} already exists (status={existing[0]})")

    node_rows, id_map = build_node_rows(atoms, version)
    edge_rows = build_edge_rows(edges, id_map, version)

    conn.execute(INSERT_VERSION, {"version": version, "notes": notes})
    conn.execute(INSERT_NODES, node_rows)
    conn.execute(INSERT_EDGES, edge_rows)
    conn.execute(FREEZE_VERSION, {"version": version})
    return {"n_nodes": len(node_rows), "n_edges": len(edge_rows), "id_map": id_map}


def _input_hashes(paths: list[Path]) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths if p.exists()
    }


def _load_chunk_ids(conn=None) -> set[str]:
    """Every chunk_id present in book_chunks — the provenance source of truth."""
    if conn is not None:
        return {row[0] for row in conn.execute(text("SELECT chunk_id FROM book_chunks")).all()}
    from bachelor_s_project_i_lms.db.session import engine

    with engine.connect() as db_conn:
        return {row[0] for row in db_conn.execute(text("SELECT chunk_id FROM book_chunks")).all()}


# ----------------------------- entry point -----------------------------

def run(
    version: int = GRAPH_VERSION,
    dry_run: bool = False,
    conn=None,
    reviewed_atoms_path: Path = REVIEWED_ATOMS_PATH,
    reviewed_edges_path: Path = REVIEWED_EDGES_PATH,
    atoms_path: Optional[Path] = None,
    final_edges_path: Optional[Path] = None,
    tree_path: Optional[Path] = CHECKPOINT_PATH,
    edits_path: Path = review.EDITS_PATH,
    report_path: Path = REPORT_PATH,
    chunk_ids: Optional[set[str]] = None,
    check_db: bool = True,
) -> dict:
    """Verify, gate and publish ``graph_v1`` (or dry-run the checks)."""
    if not reviewed_atoms_path.exists() or not reviewed_edges_path.exists():
        raise RuntimeError("missing reviewed_atoms/reviewed_edges — run Phase 7 first")
    reviewed_atoms = json.loads(reviewed_atoms_path.read_text(encoding="utf-8"))
    reviewed_edges = json.loads(reviewed_edges_path.read_text(encoding="utf-8"))

    atoms = load_atoms(atoms_path) if atoms_path is not None else load_atoms()
    edges_file = final_edges_path or FINAL_EDGES_PATH
    if atoms is None or not edges_file.exists():
        raise RuntimeError("missing atoms.json/final_edges.json — cannot verify staleness")
    final_edges = json.loads(edges_file.read_text(encoding="utf-8"))
    edits = json.loads(edits_path.read_text(encoding="utf-8")) if edits_path.exists() else {}

    tree = json.loads(tree_path.read_text(encoding="utf-8")) if tree_path and tree_path.exists() else None
    tree_roots = [r["title"] for r in tree] if tree is not None else None

    warnings: list[str] = []
    if chunk_ids is None and check_db:
        try:
            chunk_ids = _load_chunk_ids(conn)
        except Exception as exc:  # DB unavailable: keep the gate useful offline
            warnings.append(f"provenance check skipped (book_chunks unavailable: {exc})")

    errors = check_staleness(reviewed_atoms, reviewed_edges, atoms, final_edges, edits)
    gate = check_gate(reviewed_atoms, reviewed_edges, chunk_ids=chunk_ids, tree_roots=tree_roots)
    errors.extend(gate["errors"])
    warnings.extend(gate["warnings"])

    report = {
        "version": version,
        "dry_run": dry_run,
        "errors": errors,
        "warnings": warnings,
        "stats": gate["stats"],
        "inputs": _input_hashes([atoms_path or ATOMS_PATH, edges_file, edits_path,
                                 reviewed_atoms_path, reviewed_edges_path]),
    }

    if errors:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        for err in errors:
            print(f"! {err}")
        raise RuntimeError("freeze gate failed — nothing was published")

    if dry_run:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"= dry run OK: {gate['stats']['n_atoms']} atoms, {gate['stats']['n_edges']} edges")
        for warning in warnings:
            print(f"! {warning}")
        return report

    notes = json.dumps({"source": "phase8", "inputs": report["inputs"]}, ensure_ascii=False)
    if conn is not None:
        published = publish(conn, reviewed_atoms, reviewed_edges, version, notes)
    else:
        from bachelor_s_project_i_lms.db.session import engine

        with engine.begin() as db_conn:
            published = publish(db_conn, reviewed_atoms, reviewed_edges, version, notes)

    report["published"] = {"n_nodes": published["n_nodes"], "n_edges": published["n_edges"]}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"= published graph_version {version}: {published['n_nodes']} nodes, "
          f"{published['n_edges']} edges, status=frozen")
    for warning in warnings:
        print(f"! {warning}")
    return report


if __name__ == "__main__":
    run()

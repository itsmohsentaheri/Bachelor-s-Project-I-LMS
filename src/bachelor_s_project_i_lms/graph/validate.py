"""Phase 6: validation of the combined edge set — all checks in code, no LLM.

Runs the spec'd gauntlet over local + cross-branch edges merged into one graph:

1. Cycle detection (Tarjan SCC). In each cycle, remove the edge with the lowest
   confidence; repeat until the graph is a DAG. Self-loops are cycles of length 1.
2. Transitive reduction: an edge u->v is redundant when some other path
   u->...->v already implies it — remove it (recorded, so Phase 7 can audit).
3. Isolated nodes (no incoming, no outgoing): confirmed independent and
   reported — discovering a missing edge is Phase 7's judgment call, not ours.
4. Syllabus coverage: every heading in the tree owns at least one atom, and
   every atom is attached to a heading.
5. Duplicate atoms with very close title+objective embeddings: flagged for
   review (merging is a human decision, never automatic).
6. Prerequisite chains longer than 8 edges: warning, not failure.

Reads ``edges.json`` + ``cross_edges.json`` + ``tree.json`` + ``atoms.json`` and
writes ``final_edges.json`` (the cleaned combined set) and
``validation_report.json``. The original checkpoints are left untouched so every
decision here stays auditable and reversible.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable, Optional

from bachelor_s_project_i_lms.graph.atom_meta import load_atoms
from bachelor_s_project_i_lms.graph.decompose import load_checkpoint
from bachelor_s_project_i_lms.graph.prereqs import load_edges
from bachelor_s_project_i_lms.graph.cross_branch import load_cross_edges

CHECKPOINTS = Path(__file__).resolve().parent / "ai-engineering" / "checkpoints"
FINAL_EDGES_PATH = CHECKPOINTS / "final_edges.json"
REPORT_PATH = CHECKPOINTS / "validation_report.json"

DUPLICATE_SIMILARITY = 0.95  # cosine of title+objective embeddings -> flag for review
CHAIN_WARNING = 8            # prerequisite chains longer than this get a warning


# ----------------------------- graph plumbing -----------------------------

def _adjacency(edges: list[dict]) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        adj[e["from_temp_id"]].append(e["to_temp_id"])
    return adj


def _sccs(nodes: list[str], edges: list[dict]) -> list[list[str]]:
    """Tarjan strongly-connected components, iterative (no recursion limit worries)."""
    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    result: list[list[str]] = []

    adj = _adjacency(edges)
    for start in nodes:
        if start in index:
            continue
        work = [(start, 0)]
        while work:
            node, pi = work[-1]
            if pi == 0:
                index[node] = lowlink[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)
                on_stack.add(node)
            advanced = False
            for i in range(pi, len(adj[node])):
                succ = adj[node][i]
                if succ not in index:
                    work[-1] = (node, i + 1)
                    work.append((succ, 0))
                    advanced = True
                    break
                if succ in on_stack:
                    lowlink[node] = min(lowlink[node], index[succ])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == index[node]:
                comp: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == node:
                        break
                result.append(comp)
    return result


def _edge_confidence(edges: list[dict], u: str, v: str) -> Optional[float]:
    for e in edges:
        if e["from_temp_id"] == u and e["to_temp_id"] == v:
            return e["confidence"]
    return None


def break_cycles(nodes: list[str], edges: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove the lowest-confidence edge from each cycle until the graph is a DAG.

    Returns (clean_edges, removed_edges) — removals keep their full record for
    the audit trail.
    """
    edges = [dict(e) for e in edges]
    removed: list[dict] = []

    def edge_key(e: dict) -> tuple:
        return (e["confidence"], e["from_temp_id"], e["to_temp_id"])

    while True:
        # a component is cyclic if it has 2+ members, or 1 member with a self-edge
        self_ids = {e["from_temp_id"] for e in edges if e["from_temp_id"] == e["to_temp_id"]}
        cyclic = [c for c in _sccs(nodes, edges) if len(c) > 1 or c[0] in self_ids]
        if not cyclic:
            break
        # one removal attempt per component: after removing the weakest edge of
        # a cycle, other cycles may dissolve too, so re-detect before next pass
        changed = False
        for comp in cyclic:
            members = set(comp)
            if len(comp) == 1:
                # singleton with a self-edge: the self-loop IS the cycle
                inside = [e for e in edges
                          if e["from_temp_id"] in members and e["to_temp_id"] in members
                          and e["from_temp_id"] == e["to_temp_id"]]
            else:
                inside = [e for e in edges
                          if e["from_temp_id"] in members and e["to_temp_id"] in members]
            weakest = min(inside, key=edge_key)
            edges.remove(weakest)
            removed.append({**weakest, "removed_reason": "cycle (lowest confidence)"})
            changed = True
        if not changed:
            break
    return edges, removed


def transitive_reduction(nodes: list[str], edges: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop every edge u->v that is already implied by a longer path u->...->v.

    DAG-only (run after break_cycles). Returns (reduced_edges, removed_edges).
    Classic reachability pruning: for each edge u->v, if v is reachable from u
    through some other node, the edge is redundant.
    """
    by_pair: dict[tuple[str, str], dict] = {}
    for e in edges:
        by_pair.setdefault((e["from_temp_id"], e["to_temp_id"]), e)

    def reachable(from_id: str, target: str, skip_pair: tuple[str, str]) -> bool:
        """True if target is reachable from from_id without the direct edge skip_pair."""
        adj: dict[str, list[str]] = defaultdict(list)
        for (u, v) in by_pair:
            if (u, v) == skip_pair:
                continue
            adj[u].append(v)
        seen = {from_id}
        queue = deque(adj[from_id])
        while queue:
            n = queue.popleft()
            if n == target:
                return True
            if n in seen:
                continue
            seen.add(n)
            queue.extend(adj[n])
        return False

    removed: list[dict] = []
    for (u, v), e in sorted(by_pair.items()):
        if reachable(u, v, (u, v)):
            removed.append({**e, "removed_reason": "transitive reduction"})
            del by_pair[(u, v)]

    reduced = sorted(by_pair.values(), key=lambda e: (e["from_temp_id"], e["to_temp_id"]))
    return reduced, removed


# ----------------------------- structural checks -----------------------------

def isolated_nodes(nodes: list[str], edges: list[dict]) -> list[str]:
    """Atoms with no incoming and no outgoing edges."""
    touched = set()
    for e in edges:
        touched.add(e["from_temp_id"])
        touched.add(e["to_temp_id"])
    return [n for n in nodes if n not in touched]


def syllabus_coverage(tree: list[dict], atoms: list[dict]) -> dict:
    """Every heading owns >= 1 atom; every atom attaches to some heading."""
    headings_without_atoms: list[str] = []
    for root in tree:
        stack = [root]
        while stack:
            node = stack.pop()
            kids = node.get("children") or []
            stack.extend(kids)
            if not node["is_atomic"] and not kids:
                headings_without_atoms.append(node["title"])

    atom_ids = {a["temp_id"] for a in atoms}

    def leaf_ids(n: dict) -> set[str]:
        if n["is_atomic"] or not (n.get("children") or []):
            return {n["temp_id"]}
        out: set[str] = set()
        for c in n["children"]:
            out |= leaf_ids(c)
        return out

    tree_leaves: set[str] = set()
    for root in tree:
        tree_leaves |= leaf_ids(root)
    unattached = sorted(atom_ids - tree_leaves)

    return {
        "headings_without_atoms": headings_without_atoms,
        "atoms_not_in_tree": unattached,
        "n_headings_checked": sum(1 for r in tree for _ in r.get("children") or []) or len(tree),
    }


def duplicate_atoms(atoms: list[dict], embed_fn: Callable[[str], list[float]]) -> list[list[dict]]:
    """Pairs of atoms whose title+objective embeddings are nearly identical.

    Flagged, never merged — merging is a Phase 7 human decision.
    """
    texts = [f"{a['title']}. {a.get('learning_objective', '')}" for a in atoms]
    vecs = [embed_fn(t) for t in texts]

    def cos(u: list[float], v: list[float]) -> float:
        num = sum(a * b for a, b in zip(u, v))
        nu = sum(a * a for a in u) ** 0.5 or 1.0
        nv = sum(b * b for b in v) ** 0.5 or 1.0
        return num / (nu * nv)

    dupes: list[list[dict]] = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if cos(vecs[i], vecs[j]) >= DUPLICATE_SIMILARITY:
                dupes.append([atoms[i], atoms[j]])
    return dupes


def longest_chain(nodes: list[str], edges: list[dict]) -> int:
    """Longest prerequisite chain (in edges) across the graph."""
    adj = _adjacency(edges)
    memo: dict[str, int] = {}

    def depth(n: str, visiting: set[str]) -> int:
        if n in memo:
            return memo[n]
        if n in visiting:
            return 0  # cycle guard; break_cycles runs first anyway
        best = 0
        for m in adj[n]:
            best = max(best, depth(m, visiting | {n}))
        memo[n] = best + (1 if adj[n] else 0)
        return memo[n]

    return max((depth(n, set()) for n in nodes), default=0)


# ----------------------------- entry point -----------------------------

def validate(
    tree: list[dict],
    atoms: list[dict],
    local_edges: list[dict],
    cross_edges: list[dict],
    embed_fn: Optional[Callable[[str], list[float]]] = None,
) -> dict:
    """Full Phase 6 gauntlet. Returns the report dict (also written to disk)."""
    nodes = [a["temp_id"] for a in atoms]
    by_id = {a["temp_id"]: a for a in atoms}
    edges = local_edges + cross_edges

    report: dict = {"n_atoms": len(atoms), "n_edges_in": len(edges)}

    # 1. cycles
    dag_edges, cycles_removed = break_cycles(nodes, edges)
    report["cycle_edges_removed"] = [
        {"from": by_id[e["from_temp_id"]]["title"], "to": by_id[e["to_temp_id"]]["title"],
         "confidence": e["confidence"], "origin": e.get("origin"), "rationale": e.get("rationale")}
        for e in cycles_removed
    ]
    report["n_edges_after_cycles"] = len(dag_edges)

    # 2. transitive reduction
    reduced_edges, redundant_removed = transitive_reduction(nodes, dag_edges)
    report["redundant_edges_removed"] = [
        {"from": by_id[e["from_temp_id"]]["title"], "to": by_id[e["to_temp_id"]]["title"],
         "confidence": e["confidence"], "origin": e.get("origin")}
        for e in redundant_removed
    ]
    report["n_edges_after_reduction"] = len(reduced_edges)

    # 3. isolated nodes: confirm independent, report for Phase 7
    iso = isolated_nodes(nodes, reduced_edges)
    report["isolated_nodes"] = [by_id[n]["title"] for n in iso]

    # 4. syllabus coverage
    report["coverage"] = syllabus_coverage(tree, atoms)

    # 5. duplicate atoms (embedding-cosine on title+objective)
    if embed_fn is not None:
        dupes = duplicate_atoms(atoms, embed_fn)
        report["duplicate_atom_pairs"] = [
            {"a": a["title"], "b": b["title"], "a_id": a["temp_id"], "b_id": b["temp_id"]}
            for a, b in dupes
        ]
    else:
        report["duplicate_atom_pairs"] = "skipped (no embed_fn)"

    # 6. chain length warning
    chain = longest_chain(nodes, reduced_edges)
    report["longest_chain_edges"] = chain
    report["chain_warning"] = chain > CHAIN_WARNING

    report["final_edges"] = reduced_edges
    return report


def run(
    embed_fn: Optional[Callable[[str], list[float]]] = None,
    tree_path: Optional[Path] = None,
    atoms_path: Optional[Path] = None,
    local_path: Optional[Path] = None,
    cross_path: Optional[Path] = None,
    final_path: Path = FINAL_EDGES_PATH,
    report_path: Path = REPORT_PATH,
) -> dict:
    """Load all checkpoints, validate, write final_edges.json + validation_report.json."""
    from bachelor_s_project_i_lms.graph.decompose import CHECKPOINT_PATH as TREE_DEFAULT
    from bachelor_s_project_i_lms.graph.atom_meta import ATOMS_PATH as ATOMS_DEFAULT
    from bachelor_s_project_i_lms.graph.prereqs import EDGES_PATH as LOCAL_DEFAULT

    tree = load_checkpoint(tree_path or TREE_DEFAULT)
    atoms = load_atoms(atoms_path or ATOMS_DEFAULT)
    local_edges = load_edges(local_path or LOCAL_DEFAULT) or []
    cross = load_cross_edges(cross_path) if cross_path is not None else load_cross_edges()
    cross_edges = cross or []
    if tree is None or atoms is None:
        raise RuntimeError("missing checkpoints — run decompose + atom_meta first")

    report = validate(tree, atoms, local_edges, cross_edges, embed_fn=embed_fn)

    final_edges = report.pop("final_edges")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(final_edges, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"= atoms: {report['n_atoms']}, edges in: {report['n_edges_in']}")
    print(f"= cycle edges removed: {len(report['cycle_edges_removed'])}")
    print(f"= redundant edges removed: {len(report['redundant_edges_removed'])}")
    print(f"= final edges: {report['n_edges_after_reduction']}")
    print(f"= isolated nodes: {report['isolated_nodes']}")
    cov = report["coverage"]
    print(f"= coverage: headings w/o atoms: {len(cov['headings_without_atoms'])}, "
          f"atoms not in tree: {len(cov['atoms_not_in_tree'])}")
    print(f"= duplicate pairs: "
          f"{len(report['duplicate_atom_pairs']) if isinstance(report['duplicate_atom_pairs'], list) else report['duplicate_atom_pairs']}")
    print(f"= longest chain: {report['longest_chain_edges']} edges"
          + ("  << WARNING > CHAIN_WARNING" if report["chain_warning"] else ""))
    return report


if __name__ == "__main__":
    run()

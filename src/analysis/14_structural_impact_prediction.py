"""Structural impact prediction — edge recommendation from structural issues.

Reads all existing outputs and for each structural issue (fragility, isolation,
k-core lock-in, value underfunding, narrative cleavage, perception isolation):

  1. Generates candidate edges that would remediate the issue
  2. Runs counterfactual simulation: what new perception/narrative pathways open?
  3. Scores candidates by governance value
  4. Outputs ranked candidates with human-readable impact statements

Outputs per platform:
  - structural_impact_candidates.csv — scored recommendations
  - structural_impact_report.json — summary with findings

Usage:
    set KTOOL_PLATFORM_ID=173 & set KTOOL_OUTPUT_SUBDIR=test
    python src/analysis/14_structural_impact_prediction.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from graph_utils import (
    ANALYSIS_DIR,
    ANALYTICS_DIR,
    DATA_DIR,
    build_graph,
    read_csv_safe,
    simple_graph,
    write_frame,
    write_json,
)


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 2:
        return pd.read_csv(path)
    return pd.DataFrame()


def label_for(nid: str, node_labels: dict[str, str]) -> str:
    return node_labels.get(nid, nid)


# ─── Structural issue types ─────────────────────────────────────────────
ISSUE_TYPES = [
    "fragility",
    "isolation",
    "kcore_exclusion",
    "value_underfunding",
    "narrative_cleavage",
    "perception_isolation",
]

BRIDGE_TYPE_NAMES = {
    "coalition_reinforcement": "Reinforces shared perception-narrative space",
    "learning_bridge": "Bridges separate perception-narrative spaces",
    "cleavage_breach": "Connects conflicting perception-narrative spaces",
    "no_immediate_impact": "Mainly structural — no near-term perception change",
}

VALUE_DIMENSION_NAMES = {
    "cultural_identity": "Cultural Identity",
    "social_justice": "Social Justice",
    "collaboration": "Collaboration",
    "innovation_drive": "Innovation Drive",
    "evidence_based": "Evidence-Based",
    "community_autonomy": "Community Autonomy",
    "austerity_scarcity": "Austerity & Scarcity",
}


# ─── Graph loading ──────────────────────────────────────────────────────

def build_full_graph() -> nx.Graph:
    """Build the richest possible graph: base edges + claim edges."""
    G_multi = build_graph(directed=False)
    G = simple_graph(G_multi) if hasattr(G_multi, "to_undirected") else nx.Graph(G_multi)

    claim_edges = load_csv(ANALYSIS_DIR / "narrative_layers" / "claim_edges.csv")
    if not claim_edges.empty:
        for _, row in claim_edges.iterrows():
            src = str(row["source_global_id"])
            tgt = str(row["target_global_id"])
            if G.has_node(src) and G.has_node(tgt):
                G.add_edge(src, tgt, weight=row.get("weight", 0.8), edge_origin="claim_edge")
    return G


def build_node_registry(G: nx.Graph) -> pd.DataFrame:
    """Build a single registry of all nodes with their metadata."""
    nodes_df, _ = read_csv_safe(ANALYTICS_DIR / "nodes.csv"), read_csv_safe(ANALYTICS_DIR / "edges.csv")
    if nodes_df.empty:
        nodes_df = read_csv_safe(DATA_DIR / "nodes.csv")

    reg = pd.DataFrame({"global_id": list(G.nodes())})
    node_type_map = dict(zip(nodes_df["global_id"].astype(str), nodes_df["node_type"])) if not nodes_df.empty else {}
    node_label_map = dict(zip(nodes_df["global_id"].astype(str), nodes_df["label"])) if not nodes_df.empty else {}
    reg["node_type"] = reg["global_id"].map(node_type_map).fillna("unknown")
    reg["label"] = reg["global_id"].map(node_label_map).fillna("")

    centrality = load_csv(ANALYSIS_DIR / "node_centrality.csv")
    if not centrality.empty:
        cent_map = centrality.set_index("global_id")["betweenness_centrality"].to_dict()
        deg_map = centrality.set_index("global_id")["degree"].to_dict()
        reg["betweenness"] = reg["global_id"].map(cent_map).fillna(0.0)
        reg["degree"] = reg["global_id"].map(deg_map).fillna(0)
    else:
        deg = dict(G.degree())
        reg["betweenness"] = 0.0
        reg["degree"] = reg["global_id"].map(deg).fillna(0)

    kcore = load_csv(ANALYSIS_DIR / "kcore_membership.csv")
    if not kcore.empty:
        kc_map = kcore.set_index("global_id")["k_core_number"].to_dict()
        reg["k_core"] = reg["global_id"].map(kc_map).fillna(0)

    comp = load_csv(ANALYSIS_DIR / "component_membership.csv")
    if not comp.empty:
        cid_map = comp.set_index("global_id")["component_id"].to_dict()
        csize_map = comp.set_index("global_id")["component_size"].to_dict()
        reg["component_id"] = reg["global_id"].map(cid_map).fillna("isolated")
        reg["component_size"] = reg["global_id"].map(csize_map).fillna(1)

    art = load_csv(ANALYSIS_DIR / "articulation_points.csv")
    if not art.empty:
        art_nodes = set(art["global_id"].astype(str))
        reg["is_articulation"] = reg["global_id"].isin(art_nodes)

    # Claim metadata
    claim_nodes = load_csv(ANALYSIS_DIR / "narrative_layers" / "claim_nodes.csv")
    if not claim_nodes.empty:
        vd_map = claim_nodes.set_index("global_id")["value_dimension"].to_dict()
        nl_map = claim_nodes.set_index("global_id")["narrative_level"].to_dict()
        reg["value_dimension"] = reg["global_id"].map(vd_map).fillna("")
        reg["narrative_level"] = reg["global_id"].map(nl_map).fillna("")

    return reg


def node_index_for(G: nx.Graph, reg: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    nid_to_idx = {nid: i for i, nid in enumerate(G.nodes())}
    idx_to_nid = {i: nid for nid, i in nid_to_idx.items()}
    return nid_to_idx, idx_to_nid


# ─── BFS utilities ─────────────────────────────────────────────────────

def bfs_perception_nodes(
    start_nodes: list[str],
    G: nx.Graph,
    reg: pd.DataFrame,
    max_hops: int = 3,
    exclude: set[str] | None = None,
) -> dict[str, set[str]]:
    """BFS from each start node up to max_hops; return sets of perception and claim nodes reachable."""
    exclude = exclude or set()
    result: dict[str, set[str]] = {}
    for start in start_nodes:
        if start not in G:
            result[start] = set()
            continue
        seen = {start} | exclude
        frontier = {start}
        reached: set[str] = set()
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for neighbor in G.neighbors(node):
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    next_frontier.add(neighbor)
                    nt = reg.loc[reg["global_id"] == neighbor, "node_type"].values
                    if len(nt) > 0 and nt[0] in ("perception", "claim"):
                        reached.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        result[start] = reached
    return result


# ─── Issue detection ────────────────────────────────────────────────────

def find_fragility_candidates(
    G: nx.Graph, reg: pd.DataFrame, top_n: int = 5
) -> list[dict]:
    """Articulation points with few redundant paths → connect to a high-degree neighbor in same component."""
    candidates = []
    art_nodes = reg[reg["is_articulation"] == True].nsmallest(top_n, "degree")
    for _, node in art_nodes.iterrows():
        nid = node["global_id"]
        if nid not in G:
            continue
        neighbors = list(G.neighbors(nid))
        # Find highest-degree neighbor that is NOT an articulation point itself
        best_target = None
        best_deg = -1
        for nb in neighbors:
            nb_art = reg.loc[reg["global_id"] == nb, "is_articulation"].values
            if len(nb_art) > 0 and nb_art[0]:
                continue
            nb_deg = reg.loc[reg["global_id"] == nb, "degree"].values
            d = int(nb_deg[0]) if len(nb_deg) > 0 else 0
            if d > best_deg:
                best_deg = d
                best_target = nb
        if best_target:
            candidates.append({
                "source": nid,
                "target": best_target,
                "issue_type": "fragility",
                "source_type": node["node_type"],
                "target_type": reg.loc[reg["global_id"] == best_target, "node_type"].values[0] if len(reg.loc[reg["global_id"] == best_target, "node_type"].values) > 0 else "unknown",
                "rationale": f"Articulation point {label_for(nid, {})} has degree {int(node['degree'])} — adding redundant path to {label_for(best_target, {})} reduces fragility",
            })
    return candidates


def find_isolation_candidates(
    G: nx.Graph, reg: pd.DataFrame, top_n: int = 5
) -> list[dict]:
    """Small components → bridge to largest component via highest-betweenness node."""
    candidates = []
    comp_size_map = dict(zip(reg["global_id"], reg["component_size"]))
    comp_id_map = dict(zip(reg["global_id"], reg["component_id"]))
    # Find the largest component
    largest_comp_nodes = reg[reg["component_id"] == reg["component_id"].mode().values[0]]["global_id"].tolist() if not reg.empty else []
    largest_set = set(largest_comp_nodes)

    # Small components (size <= 5) that are NOT the largest component
    small_comp_ids = reg[reg["component_id"] != reg["component_id"].mode().values[0] if not reg.empty else ""].groupby("component_id")["global_id"].apply(list).to_dict()

    # Actually simpler: filter by component_size <= 5
    small_nodes = reg[
        (reg["component_size"] <= 5)
        & (reg["component_size"] > 0)
        & (~reg["global_id"].isin(largest_set))
    ]
    # Take the highest-betweenness node per small component
    grouped = small_nodes.sort_values("betweenness", ascending=False).groupby("component_id").head(1)
    for _, node in grouped.head(top_n).iterrows():
        nid = node["global_id"]
        # Find closest node in largest component (shortest path in whole graph)
        # Fallback: use lowest degree centrality difference
        if not largest_set:
            continue
        candidates.append({
            "source": nid,
            "target": list(largest_set)[0],
            "issue_type": "isolation",
            "source_type": node["node_type"],
            "target_type": reg.loc[reg["global_id"] == list(largest_set)[0], "node_type"].values[0] if len(reg.loc[reg["global_id"] == list(largest_set)[0], "node_type"].values) > 0 else "unknown",
            "rationale": f"Node {label_for(nid, {})} sits in a small component (size {int(node['component_size'])}) — bridging to the main component reduces isolation",
        })
    return candidates


def find_kcore_exclusion_candidates(
    G: nx.Graph, reg: pd.DataFrame, top_n: int = 5
) -> list[dict]:
    """Peripheral high-betweenness nodes → connect to a core node."""
    candidates = []
    if "k_core" not in reg.columns:
        return candidates
    max_k = reg["k_core"].max()
    # Nodes with k_core <= 2 but high betweenness
    peripheral = reg[(reg["k_core"] <= 2) & (reg["betweenness"] > 0.01)].nsmallest(top_n, "k_core")
    core_nodes = reg[reg["k_core"] >= max_k * 0.7]
    for _, node in peripheral.iterrows():
        nid = node["global_id"]
        # Find highest-betweenness core node
        best_core = core_nodes.sort_values("betweenness", ascending=False).iloc[0] if not core_nodes.empty else None
        if best_core is None:
            continue
        target = best_core["global_id"]
        candidates.append({
            "source": nid,
            "target": target,
            "issue_type": "kcore_exclusion",
            "source_type": node["node_type"],
            "target_type": best_core["node_type"],
            "rationale": f"Node {label_for(nid, {})} has k-core={int(node['k_core'])} with high betweenness ({node['betweenness']:.4f}) but sits outside the core — linking to core node {label_for(target, {})} (k-core={int(best_core['k_core'])}) brings it into the influential coalition",
        })
    return candidates


def find_value_underfunding_candidates(
    G: nx.Graph, reg: pd.DataFrame, top_n: int = 5
) -> list[dict]:
    """Claims in underfunded value dimensions → connect to well-funded projects/agents."""
    candidates = []
    # Read financial-perception bridge if available
    fin_bridge = load_csv(ANALYSIS_DIR / "financial_perception_bridge.csv")
    budget_by_value: dict[str, float] = {}
    if not fin_bridge.empty and "budget" in fin_bridge.columns:
        for _, row in fin_bridge.iterrows():
            vd = str(row.get("value_dimension", "")).strip()
            b = float(row.get("budget", 0))
            if vd and vd != "nan":
                budget_by_value[vd] = budget_by_value.get(vd, 0) + b

    # Determine underfunded dimensions
    if budget_by_value:
        min_budget = min(budget_by_value.values())
        underfunded = {vd for vd, b in budget_by_value.items() if b <= min_budget + 1}
    else:
        # Fallback: use claim count as proxy (less counts = less attention)
        vd_counts = reg[reg["value_dimension"] != ""]["value_dimension"].value_counts()
        min_count = vd_counts.min() if not vd_counts.empty else 0
        underfunded = set(vd_counts[vd_counts <= min_count].index)

    # Find claims in underfunded dimensions
    underfunded_claims = reg[reg["value_dimension"].isin(underfunded) & (reg["node_type"] == "claim")]
    # Find well-funded initiatives (projects with high degree)
    well_funded = reg[(reg["node_type"].isin(("project", "pilot", "prototype")))].nlargest(top_n, "degree")

    for _, claim_node in underfunded_claims.head(top_n).iterrows():
        cid = claim_node["global_id"]
        for _, proj in well_funded.iterrows():
            pid = proj["global_id"]
            if pid == cid:
                continue
            vd_name = VALUE_DIMENSION_NAMES.get(claim_node["value_dimension"], claim_node["value_dimension"])
            candidates.append({
                "source": cid,
                "target": pid,
                "issue_type": "value_underfunding",
                "source_type": "claim",
                "target_type": proj["node_type"],
                "rationale": f"Claim '{label_for(cid, {})}' expresses '{vd_name}' which has low budget/resource allocation — linking to {label_for(pid, {})} creates a narrative-to-resource pipeline",
            })
            if len([c for c in candidates if c["issue_type"] == "value_underfunding"]) >= top_n:
                break
    return candidates


def find_narrative_cleavage_candidates(
    G: nx.Graph, reg: pd.DataFrame, top_n: int = 5
) -> list[dict]:
    """Claims with different value dimensions that are NOT connected → bridge them."""
    candidates = []
    claims = reg[reg["node_type"] == "claim"]
    if claims.empty or "value_dimension" not in claims.columns:
        return candidates

    value_groups = claims.groupby("value_dimension")["global_id"].apply(list).to_dict()
    vd_pairs = list(value_groups.keys())
    for i, vd1 in enumerate(vd_pairs):
        if not vd1:
            continue
        for vd2 in vd_pairs[i + 1:]:
            if not vd2:
                continue
            if vd1 == vd2:
                continue
            # Connect one claim from each group
            c1 = value_groups[vd1][0]
            c2 = value_groups[vd2][0]
            if len(value_groups[vd1]) > 0 and len(value_groups[vd2]) > 0:
                vd1_name = VALUE_DIMENSION_NAMES.get(vd1, vd1)
                vd2_name = VALUE_DIMENSION_NAMES.get(vd2, vd2)
                candidates.append({
                    "source": c1,
                    "target": c2,
                    "issue_type": "narrative_cleavage",
                    "source_type": "claim",
                    "target_type": "claim",
                    "rationale": f"Claim in '{vd1_name}' narrative and claim in '{vd2_name}' narrative are disconnected — bridging them may create a cross-value dialogue pathway",
                })
            if len(candidates) >= top_n:
                return candidates

    return candidates


def find_perception_isolation_candidates(
    G: nx.Graph, reg: pd.DataFrame, top_n: int = 5
) -> list[dict]:
    """Perception nodes with few connections → connect to nearest high-centrality project/agent."""
    candidates = []
    perception_nodes = reg[reg["node_type"] == "perception"]
    if perception_nodes.empty:
        return candidates

    high_centrality = reg[(reg["node_type"].isin(("project", "agent")))].nlargest(top_n, "degree")

    for _, perc in perception_nodes.iterrows():
        pid = perc["global_id"]
        deg = int(perc["degree"]) if "degree" in perc else 0
        if deg >= 3:
            continue  # Already well-connected
        for _, target in high_centrality.iterrows():
            tid = target["global_id"]
            if tid == pid:
                continue
            candidates.append({
                "source": pid,
                "target": tid,
                "issue_type": "perception_isolation",
                "source_type": "perception",
                "target_type": target["node_type"],
                "rationale": f"Perception node '{label_for(pid, {})}' has only {deg} connections — linking to high-centrality {target['node_type']} '{label_for(tid, {})}' amplifies its narrative reach",
            })
            break

    return candidates


# ─── Counterfactual simulation ──────────────────────────────────────────

def simulate_impact(
    candidate: dict,
    G: nx.Graph,
    reg: pd.DataFrame,
    nid_to_idx: dict[str, int],
) -> dict:
    """Pre/post counterfactual: what perception/narrative pathways does the new edge open?"""
    src = candidate["source"]
    tgt = candidate["target"]

    # Pre-state: perception/claim nodes reachable from src and tgt individually
    src_pre = bfs_perception_nodes([src], G, reg)
    tgt_pre = bfs_perception_nodes([tgt], G, reg)
    src_reached = src_pre.get(src, set())
    tgt_reached = tgt_pre.get(tgt, set())

    # Value dimensions present in each side's neighborhood
    src_values = set()
    tgt_values = set()
    if "value_dimension" in reg.columns:
        for nid in src_reached:
            vals = reg.loc[reg["global_id"] == nid, "value_dimension"].values
            if len(vals) > 0 and vals[0]:
                src_values.add(vals[0])
        for nid in tgt_reached:
            vals = reg.loc[reg["global_id"] == nid, "value_dimension"].values
            if len(vals) > 0 and vals[0]:
                tgt_values.add(vals[0])

    # Post-state: temporarily add edge, BFS from src can now pass through tgt (and vice versa)
    G_temp = G.copy()
    G_temp.add_edge(src, tgt)

    # From src through tgt: nodes reachable from src that include tgt's pre-reached set
    src_post = bfs_perception_nodes([src], G_temp, reg)
    tgt_post = bfs_perception_nodes([tgt], G_temp, reg)
    src_reached_post = src_post.get(src, set())
    tgt_reached_post = tgt_post.get(tgt, set())

    # New nodes reachable via the new edge
    new_src = src_reached_post - src_reached
    new_tgt = tgt_reached_post - tgt_reached
    new_total = new_src | new_tgt

    # Value dimensions in newly reachable nodes
    new_values = set()
    if "value_dimension" in reg.columns:
        for nid in new_total:
            vals = reg.loc[reg["global_id"] == nid, "value_dimension"].values
            if len(vals) > 0 and vals[0]:
                new_values.add(vals[0])

    # Bridge type
    common_values = src_values & tgt_values
    if not new_total:
        bridge_type = "no_immediate_impact"
    elif new_values and not (new_values & src_values):
        bridge_type = "learning_bridge"
    elif new_values and (new_values & src_values):
        bridge_type = "coalition_reinforcement"
    else:
        common_source = src_values & set(
            reg.loc[reg["global_id"].isin(new_total), "value_dimension"].values
            if "value_dimension" in reg.columns else []
        )
        if common_source:
            bridge_type = "coalition_reinforcement"
        else:
            bridge_type = "learning_bridge"

    return {
        "source_perception_count_pre": len(src_reached),
        "target_perception_count_pre": len(tgt_reached),
        "source_perception_count_post": len(src_reached_post),
        "target_perception_count_post": len(tgt_reached_post),
        "new_perception_nodes": len(new_total),
        "new_perception_node_ids": ";".join(sorted(new_total)) if new_total else "",
        "src_value_dimensions": ";".join(sorted(src_values)) if src_values else "",
        "tgt_value_dimensions": ";".join(sorted(tgt_values)) if tgt_values else "",
        "new_value_dimensions": ";".join(sorted(new_values)) if new_values else "",
        "bridge_type": bridge_type,
    }


# ─── Scoring ────────────────────────────────────────────────────────────

def score_candidates(
    candidates: list[dict],
    simulations: list[dict],
    reg: pd.DataFrame,
) -> pd.DataFrame:
    """Compute composite governance score for each candidate."""
    rows = []
    for cand, sim in zip(candidates, simulations):
        # Normalized new perception/claim reachability
        max_reach = max(s["new_perception_nodes"] for s in simulations) if simulations else 1
        reach_score = sim["new_perception_nodes"] / max(1, max_reach)

        # New value dimensions bridged
        new_vds = len([v for v in sim.get("new_value_dimensions", "").split(";") if v])
        max_vd = max(len([v for v in s.get("new_value_dimensions", "").split(";") if v]) for s in simulations) if simulations else 1
        vd_score = new_vds / max(1, max_vd)

        # Bridge type score
        bridge_scores = {
            "learning_bridge": 1.0,
            "coalition_reinforcement": 0.6,
            "cleavage_breach": 0.8,
            "no_immediate_impact": 0.2,
        }
        bridge_score = bridge_scores.get(sim["bridge_type"], 0.2)

        # Structural issue priority
        issue_priority = {
            "fragility": 0.9,
            "perception_isolation": 0.9,
            "isolation": 0.8,
            "kcore_exclusion": 0.7,
            "value_underfunding": 0.7,
            "narrative_cleavage": 0.6,
        }
        priority = issue_priority.get(cand["issue_type"], 0.5)

        composite = reach_score * 0.25 + vd_score * 0.25 + bridge_score * 0.25 + priority * 0.25

        rows.append({
            "source_global_id": cand["source"],
            "target_global_id": cand["target"],
            "source_node_type": cand["source_type"],
            "target_node_type": cand["target_type"],
            "issue_type": cand["issue_type"],
            "bridge_type": sim["bridge_type"],
            "bridge_type_label": BRIDGE_TYPE_NAMES.get(sim["bridge_type"], sim["bridge_type"]),
            "new_perception_nodes": sim["new_perception_nodes"],
            "new_value_dimensions": sim["new_value_dimensions"],
            "reachability_score": round(reach_score, 4),
            "value_flow_score": round(vd_score, 4),
            "bridge_type_score": round(bridge_score, 4),
            "issue_priority": round(priority, 4),
            "composite_governance_score": round(composite, 4),
            "rationale": cand["rationale"],
        })
    return pd.DataFrame(rows).sort_values("composite_governance_score", ascending=False)


# ─── Impact statement generator ─────────────────────────────────────────

def generate_impact_statement(row: dict, reg: pd.DataFrame) -> str:
    """Generate a human-readable impact statement for a candidate."""
    src_label = label_for(row["source_global_id"], dict(zip(reg["global_id"], reg["label"])))
    tgt_label = label_for(row["target_global_id"], dict(zip(reg["global_id"], reg["label"])))

    parts = [f"Connecting **{src_label}** ({row['source_node_type']}) to **{tgt_label}** ({row['target_node_type']})"]

    if row["issue_type"] == "fragility":
        parts.append("addresses a structural fragility — the source is an articulation point whose removal would fragment the network.")
    elif row["issue_type"] == "isolation":
        parts.append("bridges an isolated cluster into the main network component.")
    elif row["issue_type"] == "kcore_exclusion":
        parts.append("brings a peripheral high-brokerage node into the core coalition.")
    elif row["issue_type"] == "value_underfunding":
        parts.append("creates a pipeline from an underfunded value dimension to a resource-connected node.")
    elif row["issue_type"] == "narrative_cleavage":
        parts.append("spans two disconnected value narratives, potentially creating a cross-belief dialogue.")
    elif row["issue_type"] == "perception_isolation":
        parts.append("connects an isolated perception to a high-centrality actor, amplifying its narrative reach.")

    if row["new_perception_nodes"] > 0:
        parts.append(f"Would open **{row['new_perception_nodes']} new perception/narrative pathways**.")
    else:
        parts.append("No immediate perception impact — the link is primarily structural.")

    if row["bridge_type"] == "learning_bridge":
        parts.append("This is a **learning bridge** — it connects value dimensions that were previously separate, enabling cross-coalition diffusion.")
    elif row["bridge_type"] == "coalition_reinforcement":
        parts.append("This **reinforces an existing perception coalition** — it connects nodes in a shared value space.")
    elif row["bridge_type"] == "cleavage_breach":
        parts.append("This **breaches a narrative cleavage** — it connects conflicting value frames, which carries both risk and dialogue potential.")

    return " ".join(parts)


# ─── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("STRUCTURAL IMPACT PREDICTION")
    print("=" * 60)

    print("\nBuilding graph...")
    G = build_full_graph()
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    reg = build_node_registry(G)
    print(f"  Registry: {len(reg)} nodes with metadata")

    # ── Generate candidates for each issue ──
    print("\n--- Identifying structural issues and generating candidates ---")
    all_candidates: list[dict] = []

    # Fragility
    fc = find_fragility_candidates(G, reg)
    all_candidates.extend(fc)
    print(f"  Fragility: {len(fc)} candidates")

    # Isolation
    ic = find_isolation_candidates(G, reg)
    all_candidates.extend(ic)
    print(f"  Isolation: {len(ic)} candidates")

    # K-core exclusion
    kc = find_kcore_exclusion_candidates(G, reg)
    all_candidates.extend(kc)
    print(f"  K-core exclusion: {len(kc)} candidates")

    # Value underfunding
    vc = find_value_underfunding_candidates(G, reg)
    all_candidates.extend(vc)
    print(f"  Value underfunding: {len(vc)} candidates")

    # Narrative cleavage
    nc = find_narrative_cleavage_candidates(G, reg)
    all_candidates.extend(nc)
    print(f"  Narrative cleavage: {len(nc)} candidates")

    # Perception isolation
    pc = find_perception_isolation_candidates(G, reg)
    all_candidates.extend(pc)
    print(f"  Perception isolation: {len(pc)} candidates")

    print(f"\n  Total candidates: {len(all_candidates)}")

    if not all_candidates:
        print("No candidates found. Exiting.")
        return

    # ── Counterfactual simulation ──
    print("\n--- Running counterfactual simulation ---")
    nid_to_idx, idx_to_nid = node_index_for(G, reg)
    simulations = []
    for cand in all_candidates:
        sim = simulate_impact(cand, G, reg, nid_to_idx)
        simulations.append(sim)

    did_open = sum(1 for s in simulations if s["new_perception_nodes"] > 0)
    print(f"  Candidates that open new perception pathways: {did_open}/{len(simulations)}")

    # ── Score ──
    print("\n--- Scoring candidates ---")
    scored = score_candidates(all_candidates, simulations, reg)
    print(f"  Top-3 by governance score:")
    for _, row in scored.head(3).iterrows():
        print(f"    {row['composite_governance_score']:.4f} — {row['issue_type']}: {row['source_global_id'][:20]} → {row['target_global_id'][:20]} ({row['bridge_type']})")

    # ── Generate impact statements ──
    print("\n--- Generating impact statements ---")
    scored["impact_statement"] = scored.apply(lambda r: generate_impact_statement(r.to_dict(), reg), axis=1)

    # ── Write outputs ──
    print("\n--- Writing outputs ---")
    write_frame(scored, "structural_impact_candidates.csv")
    print(f"  → structural_impact_candidates.csv ({len(scored)} candidates)")

    report = {
        "platform_id": os.environ.get("KTOOL_PLATFORM_ID", "?"),
        "output_subdir": os.environ.get("KTOOL_OUTPUT_SUBDIR", "test"),
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "total_candidates": len(all_candidates),
        "candidates_with_impact": did_open,
        "impact_summary": {
            issue: {
                "count": len([c for c in all_candidates if c["issue_type"] == issue]),
                "with_impact": len([c for c, s in zip(all_candidates, simulations) if c["issue_type"] == issue and s["new_perception_nodes"] > 0]),
            }
            for issue in ISSUE_TYPES
        },
        "top_recommendations": [],
    }

    for _, row in scored.head(5).iterrows():
        report["top_recommendations"].append({
            "source": row["source_global_id"],
            "target": row["target_global_id"],
            "source_type": row["source_node_type"],
            "target_type": row["target_node_type"],
            "issue_type": row["issue_type"],
            "bridge_type": row["bridge_type"],
            "new_perception_nodes": int(row["new_perception_nodes"]),
            "governance_score": float(row["composite_governance_score"]),
            "impact_statement": row["impact_statement"],
        })

    write_json(ANALYSIS_DIR / "structural_impact_report.json", report)
    print(f"  → structural_impact_report.json")

    print("\nDone.")


if __name__ == "__main__":
    main()

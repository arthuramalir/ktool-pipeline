"""Structural Change Possibility — leverage points, blockages, and plasticity.

Reads existing analysis outputs (centrality, robustness, k-core, GNN predictions,
intervention simulations) and asks: given this network topology, is structural
change feasible? Where would it propagate? What blocks it?

Outputs a summary JSON and supporting tables for the dashboard.

Usage:
    set KTOOL_PLATFORM_ID=173 & set KTOOL_OUTPUT_SUBDIR=test
    python src/analysis/20_structural_change_possibility.py
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from graph_utils import (
    ANALYSIS_DIR,
    ANALYTICS_DIR,
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


def main() -> None:
    print("Loading graph...")
    G_multi = build_graph(directed=False)
    G = simple_graph(G_multi)

    if G.number_of_nodes() == 0:
        print("ERROR: empty graph.")
        return

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Load existing analysis outputs ──────────────────────────────────────
    centrality = load_csv(ANALYSIS_DIR / "node_centrality.csv")
    kcore = load_csv(ANALYSIS_DIR / "kcore_membership.csv")
    components_df = load_csv(ANALYSIS_DIR / "component_membership.csv")
    robustness = load_csv(ANALYSIS_DIR / "robustness_decay.csv")
    top_fragility = load_csv(ANALYSIS_DIR / "top_fragility_nodes.csv")
    bridge_agents = load_csv(ANALYSIS_DIR / "bridge_agents.csv")
    vulnerable = load_csv(ANALYSIS_DIR / "vulnerable_connectors.csv")
    intervention = load_csv(ANALYSIS_DIR / "link_intervention_scores.csv")
    gnn_recs = load_csv(ANALYSIS_DIR / "gnn" / "gnn_link_recommendations.csv")
    structural_summary = ANALYSIS_DIR / "structural_summary.json"
    robustness_summary = ANALYSIS_DIR / "robustness_summary.json"

    nodes_df, edges_df = read_csv_safe(ANALYTICS_DIR / "nodes.csv"), read_csv_safe(ANALYTICS_DIR / "edges.csv")
    if nodes_df.empty:
        nodes_df = read_csv_safe(ANALYSIS_DIR / ".." / "nodes.csv")
    if edges_df.empty:
        edges_df = read_csv_safe(ANALYSIS_DIR / ".." / "edges.csv")

    node_types = dict(zip(nodes_df["global_id"].astype(str), nodes_df["node_type"])) if not nodes_df.empty else {}

    # ── 1. LEVERAGE POINTS ──────────────────────────────────────────────────
    # Nodes with high betweenness = potential change amplifiers
    if not centrality.empty:
        top_betweenness = centrality.nlargest(10, "betweenness_centrality")[
            ["global_id", "betweenness_centrality", "degree"]
        ].copy()
    else:
        top_betweenness = pd.DataFrame()

    if not bridge_agents.empty:
        top_brokers = bridge_agents.nlargest(10, "score")[
            ["global_id", "score"]
        ].copy()
    else:
        top_brokers = pd.DataFrame()

    # ── 2. PLASTICITY — capacity to accept new connections ──────────────────
    # Count nodes with degree 1 or 2 (peripheral, easy to rewire)
    degrees = dict(G.degree())
    peripheral = [n for n, d in degrees.items() if d <= 2]
    medium = [n for n, d in degrees.items() if 2 < d <= 5]

    # GNN-proposed links that would merge disconnected components
    n_merging_links = 0
    if not intervention.empty and "merges_components" in intervention.columns:
        n_merging_links = int(intervention["merges_components"].sum())

    # Total GNN-predicted high-confidence links
    n_high_conf_links = 0
    if not gnn_recs.empty and "link_probability" in gnn_recs.columns:
        n_high_conf_links = int((gnn_recs["link_probability"] >= 0.7).sum())

    # ── 3. STRUCTURAL BLOCKAGES ─────────────────────────────────────────────
    # Fragile connectors (articulation points with bridge edges)
    n_vulnerable = len(vulnerable) if not vulnerable.empty else 0

    # Isolated nodes (degree 0)
    isolates = list(nx.isolates(G))
    n_isolates = len(isolates)

    # Small components (<= 3 nodes) — hard to propagate change through them
    components = list(nx.connected_components(G))
    small_components = [c for c in components if len(c) <= 3]
    n_small_components = len(small_components)
    nodes_in_small_components = sum(len(c) for c in small_components)

    # Perceptions with no path to any information node
    perception_ids = {n for n, t in node_types.items() if t == "perception"}
    information_ids = {n for n, t in node_types.items() if t == "information"}
    blocked_perceptions = []
    if perception_ids and information_ids:
        for p in perception_ids:
            if p not in G:
                blocked_perceptions.append(p)
                continue
            has_path = any(nx.has_path(G, p, info) for info in information_ids if info in G)
            if not has_path:
                blocked_perceptions.append(p)

    # ── 4. PATH DEPENDENCY / LOCK-IN ────────────────────────────────────────
    # k-core: what fraction of nodes are in the densest layer?
    kcore_numbers = nx.core_number(G)
    if kcore_numbers:
        max_k = max(kcore_numbers.values())
        top_k = [n for n, k in kcore_numbers.items() if k == max_k]
        fraction_in_dense_core = len(top_k) / G.number_of_nodes()
    else:
        max_k = 0
        fraction_in_dense_core = 0.0

    # Robustness gap (targeted vs random drop after N removals)
    targeted_final = random_final = 0.0
    if not robustness.empty and len(robustness) > 1:
        targeted_final = robustness["targeted_attack_core_residual_share"].iloc[-1]
        random_final = robustness["random_failure_core_residual_share"].iloc[-1]
    robustness_gap = targeted_final - random_final

    # Does the graph have a giant component?
    if components:
        largest = max(components, key=len)
        giant_share = len(largest) / G.number_of_nodes()
    else:
        giant_share = 0.0

    # ── 5. CHANGE READINESS SCORING ─────────────────────────────────────────
    # Composite scores (0-1, higher = more ready for structural change)
    leverage_score = min(
        (len(top_betweenness) / 10) * (1 - n_isolates / max(G.number_of_nodes(), 1)),
        1.0,
    )
    plasticity_score = min(
        (len(medium) / max(G.number_of_nodes(), 1)) * 2
        + (n_high_conf_links / max(n_high_conf_links + 1, 1)) * 0.3,
        1.0,
    )
    blockage_score = min(
        (n_vulnerable / max(G.number_of_nodes(), 1)) * 5
        + (nodes_in_small_components / max(G.number_of_nodes(), 1))
        + (len(blocked_perceptions) / max(len(perception_ids), 1)) * 0.5,
        1.0,
    )
    lockin_score = min(
        fraction_in_dense_core * 1.5
        + max(0, -robustness_gap) * 2,
        1.0,
    )

    change_readiness = {
        "leverage_score": round(leverage_score, 3),
        "plasticity_score": round(plasticity_score, 3),
        "blockage_score": round(blockage_score, 3),
        "lockin_score": round(lockin_score, 3),
        "overall_readiness": round(
            1.0 - (
                (1 - leverage_score) * 0.25
                + (1 - plasticity_score) * 0.25
                + blockage_score * 0.3
                + lockin_score * 0.2
            ),
            3,
        ),
    }

    # ── 6. NARRATIVE: what does this mean? ──────────────────────────────────
    narratives = []
    if leverage_score > 0.5:
        narratives.append("Strong leverage points exist — change could be amplified through existing bridges.")
    else:
        narratives.append("Few leverage points — change may need new bridges, not existing ones.")

    if plasticity_score > 0.5:
        narratives.append("Network has spare capacity for new connections.")
    else:
        narratives.append("Network is saturated — new links may have limited effect without pruning.")

    if blockage_score > 0.3:
        narratives.append(
            f"Blockages detected: {n_vulnerable} fragile connectors, "
            f"{nodes_in_small_components} nodes in small components, "
            f"{len(blocked_perceptions)} perceptions unreachable from information."
        )

    if lockin_score > 0.5:
        narratives.append("High path dependency — dense core resists restructuring. Change may need to start at periphery.")
    else:
        narratives.append("Low lock-in — topology is open to reconfiguration.")

    if giant_share > 0.7:
        narratives.append("Giant component connects most of the network — change can propagate widely.")
    else:
        narratives.append("Fragmented network — change may stay contained within components.")

    if robustness_gap < -0.2:
        narratives.append(
            f"Targeted removal (Δ={targeted_final:.2f}) vs random (Δ={random_final:.2f}) — "
            "network depends heavily on a few hubs. Protect them for stability, or diversify for change-readiness."
        )

    # ── 7. WRITE OUTPUT ─────────────────────────────────────────────────────
    output = {
        "graph_summary": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "components": len(components),
            "giant_component_share": round(giant_share, 3),
            "max_k_core": max_k,
            "fraction_in_dense_core": round(fraction_in_dense_core, 3),
        },
        "leverage_points": {
            "n_high_betweenness_nodes": len(top_betweenness),
            "n_bridge_agents": len(top_brokers),
            "top_betweenness_nodes": top_betweenness.to_dict("records") if not top_betweenness.empty else [],
            "top_bridge_agents": top_brokers.to_dict("records") if not top_brokers.empty else [],
        },
        "plasticity": {
            "n_peripheral_nodes_deg_1_2": len(peripheral),
            "n_medium_nodes_deg_3_5": len(medium),
            "n_gnn_high_conf_new_links": n_high_conf_links,
            "n_links_that_merge_components": n_merging_links,
        },
        "blockages": {
            "n_fragile_connectors": n_vulnerable,
            "n_isolated_nodes": n_isolates,
            "n_small_components": n_small_components,
            "nodes_in_small_components": nodes_in_small_components,
            "n_blocked_perceptions": len(blocked_perceptions),
            "blocked_perception_sample": blocked_perceptions[:5],
        },
        "path_dependency": {
            "robustness_gap_targeted_vs_random": round(robustness_gap, 4),
            "targeted_final_residual": round(targeted_final, 4),
            "random_final_residual": round(random_final, 4),
        },
        "change_readiness_scores": change_readiness,
        "narrative": narratives,
    }

    write_json(ANALYSIS_DIR / "structural_change_possibility.json", output)

    # Also write flattened tables for dashboard
    records = []
    for nid, k in sorted(kcore_numbers.items(), key=lambda x: -x[1])[:20]:
        records.append({
            "global_id": nid,
            "node_type": node_types.get(nid, "unknown"),
            "k_core": k,
            "is_peripheral": "yes" if degrees.get(nid, 0) <= 2 else "no",
        })
    write_frame(pd.DataFrame(records), "change_readiness_nodes.csv")

    print("\nChange Readiness Assessment")
    print("=" * 50)
    for k, v in change_readiness.items():
        print(f"  {k}: {v}")
    print()
    for line in narratives:
        print(f"  • {line}")
    print(f"\nOutput: {ANALYSIS_DIR / 'structural_change_possibility.json'}")
    print("Done.")


if __name__ == "__main__":
    main()

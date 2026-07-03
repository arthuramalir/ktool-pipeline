from __future__ import annotations

import json
import pandas as pd
import networkx as nx

from graph_utils import (
    ANALYSIS_DIR,
    INITIATIVE_TYPES,
    build_graph,
    enrich_with_node_metadata,
    simple_graph,
    write_frame,
    write_json,
)


def run_investment_opportunity() -> None:
    print("Initializing Day 6: Investment Opportunity Metric Calculation...")
    
    G_multi = build_graph(directed=False)
    G = simple_graph(G_multi)
    
    # 1. Calculate Edge Brokerage Defensibility from Day 5 outputs
    # High-betweenness agents are our human bridges
    betweenness = nx.betweenness_centrality(G)
    degree = nx.degree_centrality(G)
    
    # 2. Extract Agent Bridge Scores
    agent_records = []
    for node, b_score in betweenness.items():
        node_type = G.nodes[node].get("node_type", "")
        if node_type == "agent":
            agent_records.append({
                "global_id": node,
                "agent_bridge_score": round(b_score, 4),
                "agent_degree_connectivity": round(degree[node], 4)
            })
    df_agent_scores = pd.DataFrame(agent_records)
    if df_agent_scores.empty:
        df_agent_scores = pd.DataFrame(columns=["global_id", "agent_bridge_score", "agent_degree_connectivity"])
    write_frame(enrich_with_node_metadata(df_agent_scores), "agent_bridge_scores.csv")

    # 3. Extract Initiative Bridge Scores (Projects/Pilots/Prototypes)
    init_records = []
    for node, b_score in betweenness.items():
        node_type = G.nodes[node].get("node_type", "")
        if node_type in INITIATIVE_TYPES:
            init_records.append({
                "global_id": node,
                "initiative_bridge_score": round(b_score, 4),
                "initiative_degree_connectivity": round(degree[node], 4)
            })
    df_init_scores = pd.DataFrame(init_records)
    if df_init_scores.empty:
        df_init_scores = pd.DataFrame(columns=["global_id", "initiative_bridge_score", "initiative_degree_connectivity"])
    write_frame(enrich_with_node_metadata(df_init_scores), "initiative_bridge_scores.csv")

    # 4. Challenge Attention Scores (Derived from perception -> challenge links)
    challenge_attention = {}
    for edge in G_multi.edges(data=True):
        e_type = edge[2].get("edge_type", "")
        if e_type == "perception_reveals_challenge":
            # Count incoming qualitative perceptions hitting each challenge
            challenge_id = edge[1]  # Target node
            challenge_attention[challenge_id] = challenge_attention.get(challenge_id, 0) + 1
            
    challenge_records = [
        {"global_id": cid, "perception_count_weight": count} 
        for cid, count in challenge_attention.items()
    ]
    df_challenge = pd.DataFrame(challenge_records)
    if not df_challenge.empty:
        df_challenge = df_challenge.sort_values(by="perception_count_weight", ascending=False)
    else:
        df_challenge = pd.DataFrame(columns=["global_id", "perception_count_weight"])
    write_frame(enrich_with_node_metadata(df_challenge), "challenge_attention.csv")

    # 5. Composite Investment Candidates Matrix
    # Merging entities that have localized structure but need connection
    # We target initiatives that are currently operational but structurally stranded
    nodes_df = enrich_with_node_metadata(df_init_scores)
    if not nodes_df.empty:
        nodes_df["investment_rationale"] = nodes_df.apply(
            lambda r: "High local connectivity but isolated from master human broker network. High optimization potential."
            if r["initiative_bridge_score"] == 0.0 and r["initiative_degree_connectivity"] > 0
            else "Active ecosystem cross-broker initiative.", axis=1
        )
    write_frame(nodes_df, "investment_candidates.csv")

    # 6. Defensive Reporting of Blocked Dashboard Indicators
    blocked_metrics = {
        "initiative_to_perception_coverage_rate": {
            "status": "BLOCKED",
            "reason": "The 'initiative_perception_links.csv' table is entirely empty in current sync.",
            "remediation": "Update frontend forms to force project managers to tag qualitative community insights."
        },
        "semantic_alignment_score": {
            "status": "BLOCKED",
            "reason": "NLP vector embeddings have not yet been generated for raw description fields.",
            "remediation": "Run sentence-transformers pipeline across nodes.csv description fields in Sprint 2."
        }
    }
    write_json(ANALYSIS_DIR / "unsupported_dashboard_metrics.json", blocked_metrics)
    print("Day 6 Indicators Compiled Successfully.")


if __name__ == "__main__":
    run_investment_opportunity()
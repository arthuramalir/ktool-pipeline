from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from graph_utils import (
    ANALYSIS_DIR,
    INITIATIVE_TYPES,
    ensure_output_dirs,
    write_frame,
)


def run_governance_interpretation() -> None:
    print("Initializing Day 5: Governance Interpretation Processing...")
    ensure_output_dirs()

    # Paths to files generated in Day 4 Structural Analysis
    centrality_path = ANALYSIS_DIR / "node_centrality.csv"
    articulation_path = ANALYSIS_DIR / "articulation_points.csv"
    bridges_path = ANALYSIS_DIR / "bridge_edges.csv"
    kcore_path = ANALYSIS_DIR / "kcore_membership.csv"
    components_path = ANALYSIS_DIR / "component_membership.csv"

    if not centrality_path.exists() or not articulation_path.exists():
        print("[ERROR] Structural files missing from analysis directory. Run Day 4 script first.")
        return

    # Load dataframes
    df_centrality = pd.read_csv(centrality_path)
    df_art = pd.read_csv(articulation_path)
    df_kcore = pd.read_csv(kcore_path)
    df_comp = pd.read_csv(components_path)
    
    # Read bridges to calculate per-node bridge counts
    df_bridges = pd.read_csv(bridges_path) if bridges_path.exists() else pd.DataFrame()

    # Create a unified master table for scoring calculations
    master = df_centrality.merge(
        df_kcore[["global_id", "k_core_number"]], on="global_id", how="left"
    ).merge(
        df_comp[["global_id", "component_id", "component_size"]], on="global_id", how="left"
    )

    # Mark articulation status
    art_set = set(df_art["global_id"].astype(str))
    master["is_articulation"] = master["global_id"].astype(str).isin(art_set)

    # Calculate how many bridge edges touch each node
    bridge_counts = {}
    if not df_bridges.empty:
        for _, row in df_bridges.iterrows():
            s, t = str(row["source_global_id"]), str(row["target_global_id"])
            bridge_counts[s] = bridge_counts.get(s, 0) + 1
            bridge_counts[t] = bridge_counts.get(t, 0) + 1
    master["bridge_edge_count"] = master["global_id"].astype(str).map(bridge_counts).fillna(0).astype(int)

    # Define confidence framework based on source data connectivity
    def assign_confidence(row: pd.Series) -> str:
        # Agents have a highly populated relationship table -> High/Medium
        if row["node_type"] == "agent":
            return "high" if row["degree_centrality"] > 0.05 else "medium"
        # Interpretive linkages or unlinked projects -> Low
        if row["node_type"] in INITIATIVE_TYPES or row["node_type"] in {"information", "channel"}:
            return "low"
        return "medium"

    master["confidence_flag"] = master.apply(assign_confidence, axis=1)

    # --- 1. TABLE: High Degree Declared Nodes ---
    high_degree = master.sort_values(by="degree_centrality", ascending=False).head(20).copy()
    high_degree["reason"] = "High direct relationship connection volume."
    high_degree["interpretation"] = "Primary ecosystem hubs; highly visible actors driving platform activity baseline."
    high_degree["score"] = high_degree["degree_centrality"].round(4)
    
    write_frame(
        high_degree[["global_id", "label", "node_type", "score", "reason", "interpretation", "confidence_flag"]],
        "high_degree_declared_nodes.csv"
    )

    # --- 2. TABLE: Bridge Agents (High Brokerage Humans) ---
    agents = master[master["node_type"] == "agent"].copy()
    agents["score"] = (agents["betweenness_centrality"] * 0.7) + (agents["is_articulation"].astype(int) * 0.3)
    bridge_agents = agents.sort_values(by="score", ascending=False).head(15).copy()
    
    reasons_agent = []
    interpretations_agent = []
    for _, r in bridge_agents.iterrows():
        if r["is_articulation"]:
            reasons_agent.append("Critical articulation bottleneck point.")
            interpretations_agent.append("Ecosystem gatekeeper. Removal fragments local human sub-networks instantly.")
        else:
            reasons_agent.append("High topological path betweenness brokerage.")
            interpretations_agent.append("Information corridor; spans the distance between different interest groups.")
            
    bridge_agents["reason"] = reasons_agent
    bridge_agents["interpretation"] = interpretations_agent
    bridge_agents["score"] = bridge_agents["score"].round(4)

    write_frame(
        bridge_agents[["global_id", "label", "node_type", "score", "reason", "interpretation", "confidence_flag"]],
        "bridge_agents.csv"
    )

    # --- 3. TABLE: Bridge Initiatives ---
    initiatives = master[master["node_type"].isin(INITIATIVE_TYPES)].copy()
    # Initiatives rely on bridge edges or local degree since their network is highly fragmented
    initiatives["score"] = (initiatives["bridge_edge_count"] * 0.5) + (initiatives["degree_centrality"] * 0.5)
    bridge_inits = initiatives.sort_values(by="score", ascending=False).head(15).copy()
    
    bridge_inits["reason"] = "Relies purely on single-point failure edges or small components."
    bridge_inits["interpretation"] = "Isolated operational pilots acting as temporary relational stepping stones."
    bridge_inits["score"] = bridge_inits["score"].round(4)

    write_frame(
        bridge_inits[["global_id", "label", "node_type", "score", "reason", "interpretation", "confidence_flag"]],
        "bridge_initiatives.csv"
    )

    # --- 4. TABLE: Vulnerable Connectors (Systemic Single-Points-of-Failure) ---
    vulnerable = master[(master["is_articulation"] == True) & (master["bridge_edge_count"] > 0)].copy()
    vulnerable["score"] = (vulnerable["betweenness_centrality"] * 0.4) + (vulnerable["bridge_edge_count"] * 0.6)
    vulnerable_sorted = vulnerable.sort_values(by="score", ascending=False).copy()
    
    vulnerable_sorted["reason"] = "Simultaneous articulation node containing multiple active bridge edges."
    vulnerable_sorted["interpretation"] = "Systemic high-fragility failure point. Holds disparate groups together single-handedly."
    vulnerable_sorted["score"] = vulnerable_sorted["score"].round(4)

    write_frame(
        vulnerable_sorted[["global_id", "label", "node_type", "score", "reason", "interpretation", "confidence_flag"]],
        "vulnerable_connectors.csv"
    )

    # --- 5. TABLE: Isolated Entities by Type ---
    # Captures things with zero degree centrality
    isolated = master[master["degree_centrality"] == 0.0].copy()
    isolated["score"] = 0.0
    isolated["reason"] = "Topological isolate (0 edges detected in current graph extraction)."
    isolated["interpretation"] = "Latent semantic context record. Awaiting text-embedding NLP linking or data sync."
    
    write_frame(
        isolated[["global_id", "label", "node_type", "score", "reason", "interpretation", "confidence_flag"]],
        "isolated_entities_by_type.csv"
    )

    print(f"Day 5 Governance Table Compilation Complete. Outputs cleanly indexed inside {ANALYSIS_DIR}")


if __name__ == "__main__":
    run_governance_interpretation()
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import networkx as nx

from graph_utils import (
    ANALYSIS_DIR,
    ensure_output_dirs,
    load_nodes_edges,
    write_json,
    write_frame,
)


def run_readiness_audit() -> None:
    """Executes a structural and statistical audit on the processed graph layers

    to classify analytical capability and capture relational silences.
    """
    print("Initializing Graph Readiness Audit...")
    ensure_output_dirs()
    
    # Load normalized datasets using structural graph_utils
    nodes, edges = load_nodes_edges()
    
    if nodes.empty:
        print("[CRITICAL] Nodes dataset is empty. Aborting audit.")
        return

    # 1. Node Counts by node_type
    node_counts = nodes["node_type"].value_counts().reset_index()
    node_counts.columns = ["node_type", "count"]
    write_frame(node_counts, "node_type_counts.csv")
    
    # 2. Edge Counts by edge_type
    if not edges.empty:
        edge_type_counts = edges["edge_type"].value_counts().reset_index()
        edge_type_counts.columns = ["edge_type", "count"]
    else:
        edge_type_counts = pd.DataFrame(columns=["edge_type", "count"])
    write_frame(edge_type_counts, "edge_type_counts.csv")

    # 3. Structural Orphan Node Rate Tracking
    if not edges.empty:
        active_edge_nodes = set(edges["source_global_id"].astype(str)).union(
            set(edges["target_global_id"].astype(str))
        )
    else:
        active_edge_nodes = set()

    nodes["is_orphan"] = ~nodes["global_id"].astype(str).isin(active_edge_nodes)
    orphan_nodes_df = nodes[nodes["is_orphan"]].copy()
    
    # Isolate and dump raw orphan nodes for pipeline debugging
    write_frame(orphan_nodes_df.drop(columns=["is_orphan"]), "orphan_nodes.csv")
    
    orphan_rates_by_type = {}
    for node_type, group in nodes.groupby("node_type"):
        total_qty = len(group)
        orphan_qty = group["is_orphan"].sum()
        orphan_rates_by_type[str(node_type)] = {
            "total_nodes": int(total_qty),
            "orphan_count": int(orphan_qty),
            "orphan_rate": float(orphan_qty / total_qty) if total_qty > 0 else 0.0
        }

    # 4. Empty Relationship Detection (Ontology Cross-Checking)
    ONTOLOGY_EDGE_TYPES = {
        "declared_interconnection",
        "information_expresses_value",
        "perception_reveals_challenge",
        "initiative_has_agent",
        "initiative_has_lead_agent",
        "initiative_mentions_perception",
        "initiative_addresses_theme"
    }
    found_edge_types = set(edges["edge_type"].unique()) if not edges.empty else set()
    empty_relation_types = sorted(list(ONTOLOGY_EDGE_TYPES - found_edge_types))

    # 5. Network Metrics Evaluation (Overall Graph vs. Edge Families Layers)
    node_type_map = nodes.set_index("global_id")["node_type"].to_dict()
    
    # Build a base unified NetworkX model to gauge overall ecosystem health
    G_unified = nx.Graph()
    G_unified.add_nodes_from(nodes["global_id"].astype(str))
    if not edges.empty:
        for _, row in edges.iterrows():
            G_unified.add_edge(str(row["source_global_id"]), str(row["target_global_id"]))
            
    overall_summary = {
        "node_count": G_unified.number_of_nodes(),
        "edge_count": G_unified.number_of_edges(),
        "component_count": nx.number_connected_components(G_unified) if G_unified.number_of_nodes() > 0 else 0,
        "graph_density": nx.density(G_unified)
    }

    # Extract isolated metrics per network layer group
    layer_metrics = {}
    if not edges.empty:
        for family_name, group_data in edges.groupby("edge_family"):
            G_layer = nx.Graph()
            G_layer.add_nodes_from(nodes["global_id"].astype(str))
            for _, row in group_data.iterrows():
                G_layer.add_edge(str(row["source_global_id"]), str(row["target_global_id"]))
            
            layer_metrics[str(family_name)] = {
                "edge_count": G_layer.number_of_edges(),
                "component_count": nx.number_connected_components(G_layer),
                "graph_density": nx.density(G_layer)
            }

    # 6. Cross-Type Relation Sparsity Matrix
    all_types = sorted(nodes["node_type"].unique())
    sparsity_matrix = pd.DataFrame(0, index=all_types, columns=all_types)
    
    if not edges.empty:
        for _, row in edges.iterrows():
            src_type = node_type_map.get(str(row["source_global_id"]), "unknown")
            tgt_type = node_type_map.get(str(row["target_global_id"]), "unknown")
            if src_type in sparsity_matrix.index and tgt_type in sparsity_matrix.columns:
                sparsity_matrix.loc[src_type, tgt_type] += 1
                
    sparsity_df = sparsity_matrix.reset_index().rename(columns={"index": "source_node_type"})
    write_frame(sparsity_df, "relation_sparsity_matrix.csv")

    # 7. Analytical Feasibility & Readiness Classification (Defense Strategy)
    readiness_classification = {}
    for edge_type in ONTOLOGY_EDGE_TYPES:
        if edge_type in empty_relation_types:
            readiness_classification[edge_type] = "blocked"
        else:
            edge_volume = int(edges[edges["edge_type"] == edge_type].shape[0])
            # Threshold constraints can be tweaked based on baseline scale
            if edge_volume > 15:
                readiness_classification[edge_type] = "supported"
            else:
                readiness_classification[edge_type] = "weakly_supported"

    # 8. Compile Comprehensive Audit Manifest
    final_report = {
        "audited_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "graph_scope_summary": overall_summary,
        "layer_family_analysis": layer_metrics,
        "orphan_node_distributions": orphan_rates_by_type,
        "undetected_ontology_relations": empty_relation_types,
        "analytical_readiness_map": readiness_classification
    }
    
    write_json(ANALYSIS_DIR / "graph_readiness_report.json", final_report)
    print(f"Audit Complete. Successfully generated 5 core report metrics files inside: {ANALYSIS_DIR}")


if __name__ == "__main__":
    run_readiness_audit()
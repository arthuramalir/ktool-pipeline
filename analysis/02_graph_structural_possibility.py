from __future__ import annotations

import json
import networkx as nx
import pandas as pd

from graph_utils import (
    ANALYSIS_DIR,
    build_graph,
    enrich_with_node_metadata,
    safe_centrality,
    simple_graph,
    write_frame,
    write_json,
)


def run_structural_possibility() -> None:
    print("Initializing Day 4: Structural Possibility Analysis...")
    
    # 1. Build an undirected, simple graph representation for structural metrics
    G_multi = build_graph(directed=False)
    G = simple_graph(G_multi)
    
    if G.number_of_nodes() == 0:
        print("[ERROR] Graph is empty. Cannot compute structural topology.")
        return

    # 2. Compute Connected Components & Topologies
    components = sorted(list(nx.connected_components(G)), key=len, reverse=True)
    total_nodes = G.number_of_nodes()
    largest_comp_size = len(components[0]) if components else 0
    largest_comp_share = float(largest_comp_size / total_nodes) if total_nodes > 0 else 0.0
    
    # Map every node to its specific component ID for tracking
    component_membership_records = []
    for comp_idx, comp_nodes in enumerate(components):
        for node_id in comp_nodes:
            component_membership_records.append({
                "global_id": node_id,
                "component_id": f"component_{comp_idx + 1}",
                "component_size": len(comp_nodes)
            })
    comp_mem_df = pd.DataFrame(component_membership_records)
    write_frame(enrich_with_node_metadata(comp_mem_df), "component_membership.csv")

    # 3. Structural Vulnerabilities: Articulation Points & Bridges
    # Articulation points find critical nodes whose removal disconnects the network
    articulation_points = sorted(list(nx.articulation_points(G)))
    art_df = pd.DataFrame([{"global_id": node, "is_articulation_point": True} for node in articulation_points])
    if art_df.empty:
        art_df = pd.DataFrame(columns=["global_id", "is_articulation_point"])
    write_frame(enrich_with_node_metadata(art_df), "articulation_points.csv")

    # Bridges identify single-point-of-failure connections
    structural_bridges = sorted(list(nx.bridges(G)))
    bridge_records = [
        {"source_global_id": u, "target_global_id": v, "is_structural_bridge": True} 
        for u, v in structural_bridges
    ]
    bridge_df = pd.DataFrame(bridge_records)
    if bridge_df.empty:
        bridge_df = pd.DataFrame(columns=["source_global_id", "target_global_id", "is_structural_bridge"])
    write_frame(bridge_df, "bridge_edges.csv")

    # 4. K-Core Coreness Profile (Ecosystem Density Layers)
    core_numbers = nx.core_number(G)
    kcore_records = [{"global_id": node, "k_core_number": score} for node, score in core_numbers.items()]
    kcore_df = pd.DataFrame(kcore_records)
    write_frame(enrich_with_node_metadata(kcore_df), "kcore_membership.csv")

    # 5. Core Centrality Suite Execution
    centrality_df = safe_centrality(G)
    write_frame(enrich_with_node_metadata(centrality_df), "node_centrality.csv")

    # 6. Isolate Identification
    isolates = sorted(list(nx.isolates(G)))

    # 7. Compile Structural Execution Manifest
    structural_summary = {
        "analyzed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "topology_metrics": {
            "total_nodes": total_nodes,
            "total_edges": G.number_of_edges(),
            "connected_components_count": len(components),
            "largest_component_node_count": largest_comp_size,
            "largest_component_population_share": largest_comp_share,
            "total_structural_articulation_points": len(articulation_points),
            "total_structural_bridges": len(structural_bridges),
            "total_isolated_nodes": len(isolates),
            "max_k_core_layer": int(max(core_numbers.values())) if core_numbers else 0
        }
    }
    
    write_json(ANALYSIS_DIR / "structural_summary.json", structural_summary)
    print(f"Day 4 Structural Analysis Complete. Matrix saved to {ANALYSIS_DIR}")


if __name__ == "__main__":
    run_structural_possibility()
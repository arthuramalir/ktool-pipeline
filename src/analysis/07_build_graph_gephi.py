from __future__ import annotations

import os
import re

import networkx as nx
import pandas as pd
from graph_utils import ANALYSIS_DIR, build_graph, simple_graph, safe_centrality


PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
PROJECT_NAME = os.environ.get("KTOOL_PROJECT_NAME", "ALC")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return slug.strip("_") or "project"


def export_to_gephi() -> None:
    print("Preparing network graph for Gephi export...")

    # 1. Build the graph defensively
    G_multi = build_graph(directed=False)
    G = simple_graph(G_multi)

    # 2. Pre-calculate metrics to pass to Gephi as analytical attributes
    core_numbers = nx.core_number(G)
    centrality_df = safe_centrality(G)

    # Convert centrality df to a quick lookup map
    centrality_lookup = centrality_df.set_index("global_id").to_dict(
        orient="index"
    )

    # 3. Inject attributes into nodes so Gephi recognizes them
    for node in G.nodes:
        # Pull basic attributes already on the node
        n_type = G.nodes[node].get("node_type", "unknown")
        label = G.nodes[node].get("label", str(node))

        # Clear out nested or missing structures that break GraphML format strings
        G.nodes[node]["label"] = str(label)
        G.nodes[node]["node_type"] = str(n_type)

        # Inject computed topology stats
        G.nodes[node]["k_core"] = int(core_numbers.get(node, 0))

        metrics = centrality_lookup.get(node, {})
        G.nodes[node]["degree_centrality"] = float(
            metrics.get("degree_centrality", 0.0)
        )
        G.nodes[node]["betweenness_centrality"] = float(
            metrics.get("betweenness_centrality", 0.0)
        )
        G.nodes[node]["closeness_centrality"] = float(
            metrics.get("closeness_centrality", 0.0)
        )

    # 4. Save as GraphML format
    output_path = ANALYSIS_DIR / (
        f"{slugify(PROJECT_NAME)}_platform_{PLATFORM_ID}_ecosystem_graph.graphml"
    )
    nx.write_graphml(G, output_path)

    print(f"🏁 Gephi-ready file successfully created at: {output_path}")
    print("You can now open Gephi, go to File -> Open, and select this file.")


if __name__ == "__main__":
    export_to_gephi()
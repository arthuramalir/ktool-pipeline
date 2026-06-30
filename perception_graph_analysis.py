from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from strapi_utils import data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare graph communities with perception clusters")
    parser.add_argument("--graph", type=Path, default=data_dir() / "graph_enriched.graphml")
    parser.add_argument("--documents", type=Path, default=data_dir() / "documents.parquet")
    parser.add_argument("--clusters", type=Path, default=data_dir() / "cluster_results.parquet")
    parser.add_argument("--output", type=Path, default=data_dir() / "actor_perception_analysis.parquet")
    args = parser.parse_args()

    graph = nx.read_graphml(args.graph)
    documents = pd.read_parquet(args.documents)
    clusters = pd.read_parquet(args.clusters)

    undirected = graph.to_undirected()
    communities = list(nx.algorithms.community.louvain_communities(undirected, seed=42))
    membership = {}
    for idx, community in enumerate(communities):
        for node in community:
            membership[str(node)] = idx

    degree = nx.degree_centrality(undirected)
    betweenness = nx.betweenness_centrality(undirected)
    cluster_summary = clusters.groupby("document_id").agg(cluster_count=("cluster_id", "nunique"), mean_confidence=("probability", "mean")).reset_index()
    doc_types = documents.set_index("id")["entity_type"].to_dict()

    rows = []
    for node, data in graph.nodes(data=True):
        entity_type = str(data.get("entity_type") or data.get("type") or doc_types.get(node, "unknown"))
        if entity_type not in {"agent", "project", "information", "interpretation", "perception", "organization", "organisation"}:
            continue
        rows.append(
            {
                "node_id": node,
                "entity_type": entity_type,
                "graph_community": membership.get(str(node), -1),
                "degree_centrality": float(degree.get(node, 0.0)),
                "betweenness_centrality": float(betweenness.get(node, 0.0)),
                "cluster_count": int(cluster_summary.set_index("document_id").loc[node, "cluster_count"] if node in cluster_summary.set_index("document_id").index else 0),
                "mean_cluster_confidence": float(cluster_summary.set_index("document_id").loc[node, "mean_confidence"] if node in cluster_summary.set_index("document_id").index else 0.0),
            }
        )

    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(f"Wrote actor/perception analysis to {args.output}")


if __name__ == "__main__":
    main()

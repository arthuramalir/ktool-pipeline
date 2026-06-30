from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd
import pyarrow.parquet as pq

from strapi_utils import data_dir


def load_embeddings(parquet_path: Path) -> dict[str, list[float]]:
    if not parquet_path.exists():
        return {}
    table = pq.read_table(parquet_path)
    frame = table.to_pandas()
    embeddings: dict[str, list[float]] = {}
    for _, row in frame.iterrows():
        embeddings[str(row["id"])] = [float(v) for v in row["embedding"]]
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach document embeddings to graph nodes")
    parser.add_argument("--nodes", type=Path, default=data_dir() / "nodes.csv")
    parser.add_argument("--edges", type=Path, default=data_dir() / "edges.csv")
    parser.add_argument("--embeddings", type=Path, default=data_dir() / "document_embeddings.parquet")
    parser.add_argument("--output-graph", type=Path, default=data_dir() / "graph_enriched.graphml")
    parser.add_argument("--output-nodes", type=Path, default=data_dir() / "nodes_enriched.csv")
    args = parser.parse_args()

    if not args.nodes.exists() or not args.edges.exists():
        raise FileNotFoundError("nodes.csv and edges.csv are required. Run src/build_graph.py first.")

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)
    embedding_map = load_embeddings(args.embeddings)

    graph = nx.MultiDiGraph()
    enriched_rows = []
    for _, row in nodes.iterrows():
        payload = row.to_dict()
        node_id = str(payload.pop("id"))
        emb = embedding_map.get(node_id)
        if emb is not None:
            payload["embedding"] = json.dumps(emb, ensure_ascii=True)
            payload["embedding_dim"] = len(emb)
        graph.add_node(node_id, **payload)
        enriched_rows.append({"id": node_id, **payload})

    for _, row in edges.iterrows():
        graph.add_edge(
            str(row["source"]),
            str(row["target"]),
            relation=row.get("relation"),
            source_type=row.get("source_type"),
            target_type=row.get("target_type"),
        )

    args.output_graph.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(enriched_rows).to_csv(args.output_nodes, index=False)
    nx.write_graphml(graph, args.output_graph)

    print(f"Wrote enriched nodes CSV: {args.output_nodes}")
    print(f"Wrote enriched graph GraphML: {args.output_graph}")


if __name__ == "__main__":
    main()

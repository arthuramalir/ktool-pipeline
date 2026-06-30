from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from strapi_utils import data_dir


def load_embeddings(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    frame = pd.read_parquet(path)
    return np.asarray(frame["embedding"].tolist(), dtype=np.float32)


def load_graph_neighbors(graph: nx.Graph, entities: list[str]) -> dict[str, list[str]]:
    neighbors: dict[str, list[str]] = {}
    for entity in entities:
        if entity in graph:
            neighbors[entity] = list(graph.neighbors(entity))[:5]
        else:
            neighbors[entity] = []
    return neighbors


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain graph-semantic fusion at the entity level")
    parser.add_argument("--documents", type=Path, default=data_dir() / "documents.parquet")
    parser.add_argument("--graph-features", type=Path, default=data_dir() / "graph_features.parquet")
    parser.add_argument("--semantic-npy", type=Path, default=data_dir() / "embeddings_sentence_transformer.npy")
    parser.add_argument("--semantic-parquet", type=Path, default=data_dir() / "document_embeddings.parquet")
    parser.add_argument("--graph", type=Path, default=data_dir() / "graph.graphml")
    parser.add_argument("--output", type=Path, default=data_dir() / "fusion_explanations.csv")
    args = parser.parse_args()

    documents = pd.read_parquet(args.documents)
    graph_features = pd.read_parquet(args.graph_features)
    semantic = load_embeddings(args.semantic_npy if args.semantic_npy.exists() else args.semantic_parquet)
    entities = documents["id"].astype(str).tolist()
    graph_matrix = graph_features.set_index("entity_id").reindex(entities)[["degree_centrality", "betweenness_centrality", "pagerank", "clustering_coefficient", "community_membership"]].fillna(0.0).astype(np.float32).to_numpy()
    semantic_matrix = StandardScaler().fit_transform(semantic.astype(np.float32))
    graph_matrix = StandardScaler().fit_transform(graph_matrix)

    graph = nx.read_graphml(args.graph).to_undirected() if args.graph.exists() else nx.Graph()
    graph_neighbors = load_graph_neighbors(graph, entities)

    semantic_centroid = semantic_matrix.mean(axis=0)
    graph_centroid = graph_matrix.mean(axis=0)
    semantic_scores = cosine_similarity(semantic_matrix, semantic_centroid.reshape(1, -1)).ravel()
    graph_scores = cosine_similarity(graph_matrix, graph_centroid.reshape(1, -1)).ravel()

    rows = []
    for idx, entity in enumerate(entities):
        semantic_mass = float(max(semantic_scores[idx], 0.0))
        graph_mass = float(max(graph_scores[idx], 0.0))
        total = semantic_mass + graph_mass + 1e-8
        semantic_pct = semantic_mass / total
        graph_pct = graph_mass / total
        if semantic_pct >= 0.67:
            interpretation = "The entity is primarily positioned by textual similarity rather than network structure."
        elif graph_pct >= 0.67:
            interpretation = "The entity is primarily positioned by network structure rather than textual similarity."
        else:
            interpretation = "The entity is jointly shaped by semantic and structural evidence."
        semantic_neighbors = np.argsort(cosine_similarity(semantic_matrix[idx].reshape(1, -1), semantic_matrix).ravel())[::-1][1:6]
        rows.append(
            {
                "entity": entity,
                "representation_source": "semantic + graph",
                "semantic_contribution_percent": semantic_pct * 100.0,
                "graph_contribution_percent": graph_pct * 100.0,
                "nearest_semantic_neighbors": json.dumps([entities[i] for i in semantic_neighbors], ensure_ascii=True),
                "nearest_graph_neighbors": json.dumps(graph_neighbors.get(entity, []), ensure_ascii=True),
                "interpretation": interpretation,
            }
        )

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote fusion explanations to {args.output}")


if __name__ == "__main__":
    main()

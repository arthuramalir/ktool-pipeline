from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from strapi_utils import data_dir, project_root


def load_documents(documents_path: Path) -> pd.DataFrame:
    if documents_path.exists():
        return pd.read_parquet(documents_path)
    raise FileNotFoundError(f"Missing documents parquet: {documents_path}")


def load_graph(graph_path: Path) -> nx.Graph:
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph file: {graph_path}")
    graph = nx.read_graphml(graph_path)
    return graph.to_undirected()


def load_semantic_embeddings(path_candidates: list[Path]) -> tuple[np.ndarray, Path]:
    for path in path_candidates:
        if path.exists():
            if path.suffix == ".npy":
                return np.load(path), path
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
                if "embedding" in frame.columns:
                    vectors = np.asarray(frame["embedding"].tolist(), dtype=np.float32)
                    return vectors, path
    raise FileNotFoundError("No semantic embedding file found in the expected locations")


def semantic_lookup(documents: pd.DataFrame, semantic_vectors: np.ndarray) -> tuple[dict[str, np.ndarray], int]:
    doc_ids = documents["id"].astype(str).tolist()
    if len(doc_ids) != len(semantic_vectors):
        raise ValueError("Document count and embedding count do not match")
    mapping = {doc_id: semantic_vectors[idx].astype(np.float32) for idx, doc_id in enumerate(doc_ids)}
    return mapping, int(semantic_vectors.shape[1])


def graph_feature_frame(graph: nx.Graph) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["entity_id", "degree_centrality", "betweenness_centrality", "pagerank", "clustering_coefficient", "community_membership"])

    simple_graph = nx.Graph(graph)
    communities = list(nx.algorithms.community.louvain_communities(simple_graph, seed=42))
    membership = {node: idx for idx, community in enumerate(communities) for node in community}
    degree = nx.degree_centrality(simple_graph)
    betweenness = nx.betweenness_centrality(simple_graph)
    pagerank = nx.pagerank(simple_graph)
    clustering = nx.clustering(simple_graph)

    rows = []
    for node in graph.nodes():
        rows.append(
            {
                "entity_id": str(node),
                "degree_centrality": float(degree.get(node, 0.0)),
                "betweenness_centrality": float(betweenness.get(node, 0.0)),
                "pagerank": float(pagerank.get(node, 0.0)),
                "clustering_coefficient": float(clustering.get(node, 0.0)),
                "community_membership": int(membership.get(node, -1)),
            }
        )
    return pd.DataFrame(rows)


def aligned_semantic_frame(documents: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    doc_ids = documents["id"].astype(str).tolist()
    if len(doc_ids) != len(embeddings):
        raise ValueError("Document count and embedding count do not match")
    rows = []
    for doc_id, vector in zip(doc_ids, embeddings):
        rows.append({"entity_id": doc_id, "semantic_embedding": vector.astype(np.float32).tolist()})
    return pd.DataFrame(rows)


def load_node_metadata(nodes_path: Path) -> pd.DataFrame:
    if nodes_path.exists():
        return pd.read_csv(nodes_path)
    return pd.DataFrame()


def make_fused_embeddings(semantic_vectors: np.ndarray, graph_vectors: np.ndarray, model: str) -> np.ndarray:
    if model == "semantic_only":
        return semantic_vectors
    if model == "graph_only":
        return graph_vectors
    if model == "concatenated":
        return np.concatenate([semantic_vectors, graph_vectors], axis=1)
    if model == "pca_projection":
        combined = np.concatenate([semantic_vectors, graph_vectors], axis=1)
        max_components = max(1, min(combined.shape[0], combined.shape[1]))
        components = max(1, min(128, max_components, max(1, combined.shape[0] - 1)))
        return PCA(n_components=components, random_state=42).fit_transform(combined)
    raise ValueError(f"Unknown fusion model: {model}")


def cosine_similarity_to_centroid(vectors: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return cosine_similarity(vectors, centroid.reshape(1, -1)).ravel()


def explanation_rows(entity_ids: list[str], semantic_vectors: np.ndarray, graph_vectors: np.ndarray) -> pd.DataFrame:
    semantic_centroid = semantic_vectors.mean(axis=0)
    graph_centroid = graph_vectors.mean(axis=0)
    semantic_scores = cosine_similarity_to_centroid(semantic_vectors, semantic_centroid)
    graph_scores = cosine_similarity_to_centroid(graph_vectors, graph_centroid)

    rows = []
    for entity_id, semantic_score, graph_score in zip(entity_ids, semantic_scores, graph_scores):
        semantic_mass = float(max(semantic_score, 0.0))
        graph_mass = float(max(graph_score, 0.0))
        total_mass = semantic_mass + graph_mass + 1e-8
        semantic_contribution = float(semantic_mass / total_mass)
        graph_contribution = float(graph_mass / total_mass)
        if semantic_contribution >= 0.67:
            interpretation = "Primarily identified by textual characteristics."
        elif graph_contribution >= 0.67:
            interpretation = "Primarily identified by network structure."
        else:
            interpretation = "Balanced contribution from semantics and structure."
        rows.append(
            {
                "entity": entity_id,
                "semantic_similarity": float(semantic_score),
                "graph_similarity": float(graph_score),
                "semantic_contribution": semantic_contribution,
                "graph_contribution": graph_contribution,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def disagreement_rows(entity_ids: list[str], semantic_vectors: np.ndarray, graph_vectors: np.ndarray) -> pd.DataFrame:
    semantic_neighbors = cosine_similarity(semantic_vectors)
    graph_neighbors = cosine_similarity(graph_vectors)
    rows = []
    for i, entity_id in enumerate(entity_ids):
        semantic_order = np.argsort(semantic_neighbors[i])[::-1][1:11]
        graph_order = np.argsort(graph_neighbors[i])[::-1][1:11]
        semantic_set = set(semantic_order.tolist())
        graph_set = set(graph_order.tolist())
        jaccard = len(semantic_set & graph_set) / max(1, len(semantic_set | graph_set))
        disagreement = float(1.0 - jaccard)
        if disagreement >= 0.65:
            label = "potential anomaly / bridge / emerging structure"
        elif disagreement >= 0.35:
            label = "partially contested representation"
        else:
            label = "stable semantic-graph alignment"
        rows.append(
            {
                "entity": entity_id,
                "semantic_neighbor_overlap": float(jaccard),
                "representation_disagreement_score": disagreement,
                "interpretation": label,
            }
        )
    return pd.DataFrame(rows)


def save_parquet_like(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse semantic and graph representations into a unified latent space")
    parser.add_argument("--documents", type=Path, default=data_dir() / "documents.parquet")
    parser.add_argument("--graph", type=Path, default=data_dir() / "graph.graphml")
    parser.add_argument("--nodes", type=Path, default=data_dir() / "nodes.csv")
    parser.add_argument("--semantic-npy", type=Path, default=data_dir() / "embeddings_sentence_transformer.npy")
    parser.add_argument("--semantic-parquet", type=Path, default=data_dir() / "document_embeddings.parquet")
    parser.add_argument("--graph-features", type=Path, default=data_dir() / "graph_features.parquet")
    parser.add_argument("--fused-output", type=Path, default=data_dir() / "fused_embeddings.parquet")
    parser.add_argument("--explanation-output", type=Path, default=data_dir() / "entity_representation_explanation.csv")
    parser.add_argument("--disagreement-output", type=Path, default=data_dir() / "semantic_graph_disagreement.csv")
    args = parser.parse_args()

    documents = load_documents(args.documents)
    graph = load_graph(args.graph)
    node_meta = load_node_metadata(args.nodes)
    semantic_vectors, semantic_source = load_semantic_embeddings([args.semantic_npy, args.semantic_parquet])
    semantic_map, semantic_dim = semantic_lookup(documents, semantic_vectors)

    graph_features = graph_feature_frame(graph)
    graph_features.to_parquet(args.graph_features, index=False)

    entity_ids = graph_features["entity_id"].astype(str).tolist()
    semantic_common = []
    semantic_coverage = []
    for entity_id in entity_ids:
        vector = semantic_map.get(entity_id)
        if vector is None:
            semantic_common.append(np.zeros((semantic_dim,), dtype=np.float32))
            semantic_coverage.append(False)
        else:
            semantic_common.append(vector)
            semantic_coverage.append(True)
    semantic_common = np.asarray(semantic_common, dtype=np.float32)
    graph_common = graph_features[["degree_centrality", "betweenness_centrality", "pagerank", "clustering_coefficient", "community_membership"]].fillna(0.0)
    graph_common = graph_common.astype(np.float32).to_numpy()

    scaler = StandardScaler()
    semantic_scaled = scaler.fit_transform(semantic_common) if len(entity_ids) else np.empty((0, semantic_common.shape[1] if semantic_common.ndim == 2 else 0))
    graph_scaled = scaler.fit_transform(graph_common) if len(entity_ids) else np.empty((0, graph_common.shape[1] if graph_common.ndim == 2 else 0))

    fused_rows = []
    model_frames = []
    for model_name in ("semantic_only", "graph_only", "concatenated", "pca_projection"):
        fused = make_fused_embeddings(semantic_scaled, graph_scaled, model_name)
        if fused.shape[0] == 0:
            continue
        if fused.ndim == 1:
            fused = fused.reshape(-1, 1)
        for entity_id, vector, has_semantic in zip(entity_ids, fused, semantic_coverage):
            fused_rows.append(
                {
                    "entity_id": entity_id,
                    "fusion_model": model_name,
                    "embedding": json.dumps([float(v) for v in vector.tolist()], ensure_ascii=True),
                    "embedding_dim": int(vector.shape[0]),
                    "semantic_coverage": bool(has_semantic),
                    "source_embedding_file": semantic_source.name,
                }
            )
        model_frames.append(pd.DataFrame({"entity_id": entity_ids, "fusion_model": model_name, "embedding_dim": [int(fused.shape[1])] * len(entity_ids)}))

    fused_frame = pd.DataFrame(fused_rows)
    save_parquet_like(args.fused_output, fused_frame)

    explanation = explanation_rows(entity_ids, semantic_scaled, graph_scaled)
    explanation.insert(1, "semantic_available", semantic_coverage)
    explanation.to_csv(args.explanation_output, index=False)

    disagreement = disagreement_rows(entity_ids, semantic_scaled, graph_scaled)
    disagreement.insert(1, "semantic_available", semantic_coverage)
    disagreement.to_csv(args.disagreement_output, index=False)

    print(f"Wrote graph features to {args.graph_features}")
    print(f"Wrote fused embeddings to {args.fused_output}")
    print(f"Wrote entity representation explanations to {args.explanation_output}")
    print(f"Wrote semantic-graph disagreement scores to {args.disagreement_output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hdbscan
import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import umap

from strapi_utils import data_dir


@dataclass(frozen=True)
class RepresentationResult:
    name: str
    matrix: np.ndarray


class FusionAutoencoder(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        hidden_dim = max(latent_dim * 2, 32)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return latent, reconstruction


def load_documents(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_graph(path: Path) -> nx.Graph:
    return nx.read_graphml(path).to_undirected()


def load_semantic_embeddings(candidates: list[Path]) -> tuple[np.ndarray, Path]:
    for path in candidates:
        if path.exists():
            if path.suffix == ".npy":
                return np.load(path), path
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
                if "embedding" in frame.columns:
                    return np.asarray(frame["embedding"].tolist(), dtype=np.float32), path
    raise FileNotFoundError("No semantic embedding file found")


def graph_features(path: Path, graph: nx.Graph) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    communities = list(nx.algorithms.community.louvain_communities(graph, seed=42))
    membership = {node: idx for idx, community in enumerate(communities) for node in community}
    rows = []
    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph)
    pagerank = nx.pagerank(graph)
    clustering = nx.clustering(graph)
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
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


class FusionProjector(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, max(latent_dim * 2, 32)),
            torch.nn.ReLU(),
            torch.nn.Linear(max(latent_dim * 2, 32), latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, max(latent_dim * 2, 32)),
            torch.nn.ReLU(),
            torch.nn.Linear(max(latent_dim * 2, 32), input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return latent, reconstruction


def representation_matrices(documents: pd.DataFrame, semantic: np.ndarray, graph_df: pd.DataFrame) -> dict[str, np.ndarray]:
    entity_ids = documents["id"].astype(str).tolist()
    graph_matrix = graph_df.set_index("entity_id").reindex(entity_ids)[["degree_centrality", "betweenness_centrality", "pagerank", "clustering_coefficient", "community_membership"]].fillna(0.0).astype(np.float32).to_numpy()
    semantic_matrix = semantic.astype(np.float32)
    early_fusion = np.concatenate([semantic_matrix, graph_matrix], axis=1)

    projector = TruncatedSVD(n_components=min(64, max(2, early_fusion.shape[1] - 1)), random_state=42)
    learned_fusion = projector.fit_transform(early_fusion).astype(np.float32)

    return {
        "semantic_only": semantic_matrix,
        "graph_only": graph_matrix,
        "early_fusion": early_fusion,
        "learned_fusion": learned_fusion,
    }


def reduce_for_quality(matrix: np.ndarray, dimension: int) -> np.ndarray:
    if matrix.shape[1] <= dimension:
        return matrix
    return TruncatedSVD(n_components=max(2, min(dimension, matrix.shape[1] - 1)), random_state=42).fit_transform(matrix)


def project_2d(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] <= 2:
        if matrix.shape[1] == 1:
            return np.column_stack([matrix[:, 0], np.zeros(len(matrix))])
        return matrix[:, :2]
    return umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, max(3, len(matrix) // 10 or 3))).fit_transform(matrix)


def knn_preservation(high_dim: np.ndarray, low_dim: np.ndarray, k: int = 10) -> float:
    if len(high_dim) <= k + 1:
        return 1.0
    high_nn = NearestNeighbors(n_neighbors=k + 1).fit(high_dim)
    low_nn = NearestNeighbors(n_neighbors=k + 1).fit(low_dim)
    high = high_nn.kneighbors(return_distance=False)[:, 1:]
    low = low_nn.kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([len(set(h).intersection(set(l))) / float(k) for h, l in zip(high, low)]))


def cluster_run(matrix: np.ndarray, algorithm: str, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    matrix = StandardScaler().fit_transform(matrix)
    if algorithm == "kmeans":
        k = min(8, max(2, int(np.sqrt(len(matrix)))))
        model = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = model.fit_predict(matrix)
        confidence = 1.0 / (1.0 + model.transform(matrix).min(axis=1))
        return labels, confidence
    if algorithm == "hdbscan":
        model = hdbscan.HDBSCAN(min_cluster_size=max(5, len(matrix) // 30 or 5), prediction_data=True)
        labels = model.fit_predict(matrix)
        confidence = getattr(model, "probabilities_", np.ones(len(matrix), dtype=np.float32))
        return labels, confidence
    if algorithm == "gmm":
        k = min(8, max(2, int(np.sqrt(len(matrix)))))
        model = GaussianMixture(n_components=k, random_state=seed)
        labels = model.fit_predict(matrix)
        confidence = model.predict_proba(matrix).max(axis=1)
        return labels, confidence
    raise ValueError(f"Unknown clustering algorithm: {algorithm}")


def stability_score(label_runs: list[np.ndarray]) -> float:
    if len(label_runs) < 2:
        return 1.0
    scores = []
    for i in range(len(label_runs)):
        for j in range(i + 1, len(label_runs)):
            scores.append(adjusted_rand_score(label_runs[i], label_runs[j]))
    return float(np.mean(scores)) if scores else 1.0


def cluster_summary(matrix: np.ndarray, algorithm: str, seeds: list[int], params_label: str) -> dict[str, Any]:
    label_runs = []
    confidence_runs = []
    for seed in seeds:
        labels, confidence = cluster_run(matrix, algorithm, seed=seed)
        label_runs.append(labels)
        confidence_runs.append(confidence)
    base_labels = label_runs[0]
    noise_percentage = float((base_labels < 0).mean() * 100.0)
    return {
        "algorithm": algorithm,
        "parameters": params_label,
        "number_clusters": int(len(set(base_labels[base_labels >= 0].tolist()))),
        "cluster_stability": stability_score(label_runs),
        "average_uncertainty": float(np.mean([1.0 - np.clip(conf, 0.0, 1.0).mean() for conf in confidence_runs])),
        "noise_percentage": noise_percentage,
    }


def learned_fusion_representation(semantic: np.ndarray, graph_matrix: np.ndarray) -> np.ndarray:
    joint = np.concatenate([semantic, graph_matrix], axis=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor(joint, dtype=torch.float32, device=device)
    latent_dim = min(64, max(16, joint.shape[1] // 2))
    model = FusionAutoencoder(input_dim=joint.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _ in range(40):
        optimizer.zero_grad()
        latent, reconstruction = model(tensor)
        loss = loss_fn(reconstruction, tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        latent, _ = model(tensor)
    return latent.cpu().numpy()


def semantic_graph_disagreement(documents: pd.DataFrame, semantic: np.ndarray, graph_matrix: np.ndarray) -> pd.DataFrame:
    semantic_nn = NearestNeighbors(n_neighbors=min(10, len(documents))).fit(StandardScaler().fit_transform(semantic))
    graph_nn = NearestNeighbors(n_neighbors=min(10, len(documents))).fit(StandardScaler().fit_transform(graph_matrix))
    semantic_idx = semantic_nn.kneighbors(return_distance=False)
    graph_idx = graph_nn.kneighbors(return_distance=False)
    rows = []
    for i, entity_id in enumerate(documents["id"].astype(str).tolist()):
        semantic_set = set(semantic_idx[i].tolist())
        graph_set = set(graph_idx[i].tolist())
        overlap = len(semantic_set & graph_set) / max(1, len(semantic_set | graph_set))
        disagreement = 1.0 - overlap
        semantic_strength = float(np.mean(np.abs(semantic[i])))
        graph_strength = float(np.mean(np.abs(graph_matrix[i])))
        if overlap >= 0.7:
            taxonomy = "semantic + graph agreement"
        elif disagreement >= 0.65:
            taxonomy = "representation conflict"
        elif semantic_strength >= graph_strength:
            taxonomy = "semantic dominant"
        else:
            taxonomy = "structural dominant"
        rows.append(
            {
                "entity": entity_id,
                "taxonomy": taxonomy,
                "semantic_neighbor_overlap": float(overlap),
                "representation_disagreement_score": float(disagreement),
            }
        )
    return pd.DataFrame(rows)


def evaluate_quality(representations: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, matrix in representations.items():
        for dimension in (2, 16, 64):
            reduced = reduce_for_quality(matrix, dimension)
            projection = project_2d(reduced)
            labels, _ = cluster_run(reduced, "kmeans")
            rows.append(
                {
                    "representation": name,
                    "dimension": int(reduced.shape[1]),
                    "knn_preservation": knn_preservation(reduced, projection, k=10),
                    "trustworthiness": float(trustworthiness(reduced, projection, n_neighbors=min(10, max(2, len(reduced) // 10 or 2)))),
                    "silhouette_score": float(silhouette_score(reduced, labels)) if len(set(labels[labels >= 0].tolist())) > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate representation learning across semantic, graph, and fusion spaces")
    parser.add_argument("--documents", type=Path, default=data_dir() / "documents.parquet")
    parser.add_argument("--graph", type=Path, default=data_dir() / "graph.graphml")
    parser.add_argument("--graph-features", type=Path, default=data_dir() / "graph_features.parquet")
    parser.add_argument("--semantic-npy", type=Path, default=data_dir() / "embeddings_sentence_transformer.npy")
    parser.add_argument("--semantic-parquet", type=Path, default=data_dir() / "document_embeddings.parquet")
    parser.add_argument("--disagreement", type=Path, default=data_dir() / "semantic_graph_disagreement.csv")
    parser.add_argument("--quality-output", type=Path, default=data_dir() / "representation_quality.csv")
    parser.add_argument("--cluster-output", type=Path, default=data_dir() / "representation_cluster_comparison.csv")
    parser.add_argument("--taxonomy-output", type=Path, default=data_dir() / "entity_representation_types.csv")
    parser.add_argument("--fused-output", type=Path, default=data_dir() / "fused_representations.parquet")
    args = parser.parse_args()

    documents = load_documents(args.documents)
    graph = load_graph(args.graph)
    semantic, semantic_source = load_semantic_embeddings([args.semantic_npy, args.semantic_parquet])
    graph_df = graph_features(args.graph_features, graph)

    entity_ids = documents["id"].astype(str).tolist()
    graph_matrix = graph_df.set_index("entity_id").reindex(entity_ids)[["degree_centrality", "betweenness_centrality", "pagerank", "clustering_coefficient", "community_membership"]].fillna(0.0).astype(np.float32).to_numpy()
    semantic_matrix = semantic.astype(np.float32)
    early_fusion = np.concatenate([semantic_matrix, graph_matrix], axis=1)
    learned_fusion = learned_fusion_representation(StandardScaler().fit_transform(semantic_matrix), StandardScaler().fit_transform(graph_matrix))

    representations = {
        "semantic_only": semantic_matrix,
        "graph_only": graph_matrix,
        "early_fusion": early_fusion,
        "learned_fusion": learned_fusion,
    }

    quality = evaluate_quality(representations)
    quality.to_csv(args.quality_output, index=False)

    cluster_rows = []
    seeds = [7, 13, 23]
    for name, matrix in representations.items():
        for algorithm in ("kmeans", "hdbscan", "gmm"):
            cluster_rows.append({"representation": name, **cluster_summary(matrix, algorithm, seeds, params_label=json.dumps({"seeds": seeds}, ensure_ascii=True))})
    pd.DataFrame(cluster_rows).to_csv(args.cluster_output, index=False)

    disagreement = semantic_graph_disagreement(documents, StandardScaler().fit_transform(semantic_matrix), StandardScaler().fit_transform(graph_matrix))
    if args.disagreement.exists():
        existing = pd.read_csv(args.disagreement)
        if {"entity", "semantic_neighbor_overlap", "representation_disagreement_score"}.issubset(existing.columns):
            disagreement = existing.merge(disagreement[["entity", "taxonomy"]], on="entity", how="left", suffixes=("_existing", ""))
            disagreement["taxonomy"] = disagreement["taxonomy"].fillna(disagreement.get("taxonomy_existing"))
            disagreement = disagreement.drop(columns=[c for c in disagreement.columns if c.endswith("_existing")], errors="ignore")
    disagreement.to_csv(args.disagreement, index=False)
    taxonomy = disagreement[["entity", "taxonomy", "semantic_neighbor_overlap", "representation_disagreement_score"]].copy()
    taxonomy.to_csv(args.taxonomy_output, index=False)

    fused_rows = []
    for name, matrix in representations.items():
        for entity_id, vector in zip(entity_ids, matrix):
            fused_rows.append(
                {
                    "entity_id": entity_id,
                    "representation": name,
                    "vector": json.dumps([float(x) for x in np.asarray(vector).ravel().tolist()], ensure_ascii=True),
                    "dimension": int(np.asarray(vector).size),
                    "source_embedding_file": semantic_source.name,
                }
            )
    pd.DataFrame(fused_rows).to_parquet(args.fused_output, index=False)

    print(f"Wrote representation quality to {args.quality_output}")
    print(f"Wrote cluster comparison to {args.cluster_output}")
    print(f"Wrote entity taxonomy to {args.taxonomy_output}")
    print(f"Wrote fused representations to {args.fused_output}")


if __name__ == "__main__":
    main()

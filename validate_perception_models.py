from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.manifold import TSNE, trustworthiness
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
import umap

from strapi_utils import data_dir


@dataclass(frozen=True)
class RunSpec:
    representation: str
    algorithm: str
    parameters: dict[str, Any]
    seed: int


def load_documents(documents_path: Path) -> pd.DataFrame:
    return pd.read_parquet(documents_path)


def load_matrix(matrix_path: Path) -> np.ndarray:
    return np.load(matrix_path)


def load_tfidf(matrix_path: Path) -> sparse.spmatrix:
    return sparse.load_npz(matrix_path)


def run_kmeans(matrix: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    model = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = model.fit_predict(matrix)
    distances = model.transform(matrix)
    confidence = 1.0 / (1.0 + distances.min(axis=1))
    return labels, confidence


def run_gmm(matrix: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    model = GaussianMixture(n_components=k, random_state=seed)
    labels = model.fit_predict(matrix)
    confidence = model.predict_proba(matrix).max(axis=1)
    return labels, confidence


def run_spectral(matrix: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    model = SpectralClustering(n_clusters=k, random_state=seed, affinity="nearest_neighbors")
    labels = model.fit_predict(matrix)
    confidence = np.ones(len(labels), dtype=np.float32)
    return labels, confidence


def run_hdbscan(matrix: np.ndarray, min_cluster_size: int, min_samples: int | None) -> tuple[np.ndarray, np.ndarray]:
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, prediction_data=True)
    labels = model.fit_predict(matrix)
    confidence = getattr(model, "probabilities_", np.ones(len(labels), dtype=np.float32))
    return labels, confidence


def cluster_count(labels: np.ndarray) -> int:
    members = {int(label) for label in labels.tolist() if int(label) >= 0}
    return len(members)


def safe_silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    unique = {int(label) for label in labels.tolist()}
    non_noise = labels >= 0
    effective = labels[non_noise]
    if len(set(effective.tolist())) < 2 or len(effective) < 3:
        return float("nan")
    try:
        return float(silhouette_score(matrix[non_noise], effective, metric="euclidean"))
    except Exception:
        return float("nan")


def prepare_representation_matrices(sentence_embeddings: np.ndarray, tfidf_matrix: sparse.spmatrix) -> dict[str, np.ndarray]:
    tfidf_svd_dims = min(50, max(2, tfidf_matrix.shape[1] - 1)) if tfidf_matrix.shape[1] > 2 else 2
    tfidf_reduced = TruncatedSVD(n_components=tfidf_svd_dims, random_state=42).fit_transform(tfidf_matrix)
    return {
        "sentence_transformer": sentence_embeddings.astype(np.float32),
        "tfidf": tfidf_reduced.astype(np.float32),
    }


def reduction_projection(matrix: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "pca":
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        return (u[:, :2] * s[:2]).astype(np.float32)
    if method == "tsne":
        perplexity = max(5, min(30, max(5, len(matrix) // 5)))
        return TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(matrix)
    n_neighbors = min(15, max(3, len(matrix) // 10 or 3))
    return umap.UMAP(n_components=2, random_state=seed, n_neighbors=n_neighbors).fit_transform(matrix)


def reduce_for_clustering(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] <= 10:
        return matrix
    components = min(30, matrix.shape[1] - 1)
    return TruncatedSVD(n_components=max(2, components), random_state=42).fit_transform(matrix)


def run_single_clustering(representation: str, algorithm: str, matrix: np.ndarray, parameters: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    clustering_matrix = reduce_for_clustering(matrix)
    if algorithm == "kmeans":
        return run_kmeans(clustering_matrix, int(parameters["k"]), seed)
    if algorithm == "gmm":
        return run_gmm(clustering_matrix, int(parameters["k"]), seed)
    if algorithm == "spectral":
        return run_spectral(clustering_matrix, int(parameters["k"]), seed)
    if algorithm == "hdbscan":
        return run_hdbscan(clustering_matrix, int(parameters["min_cluster_size"]), parameters.get("min_samples"))
    raise ValueError(f"Unknown algorithm: {algorithm}")


def parameter_grid() -> list[tuple[str, dict[str, Any], list[int]]]:
    seeds = [7, 13, 23, 37]
    grid: list[tuple[str, dict[str, Any], list[int]]] = []
    for k in range(3, 9):
        grid.append(("kmeans", {"k": k}, seeds))
        grid.append(("gmm", {"k": k}, seeds))
        grid.append(("spectral", {"k": k}, seeds[:3]))
    for min_cluster_size in (5, 10, 15):
        for min_samples in (None, 5):
            grid.append(("hdbscan", {"min_cluster_size": min_cluster_size, "min_samples": min_samples}, [0]))
    return grid


def params_key(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, ensure_ascii=True)


def best_label_overlap(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    valid = (labels_a >= 0) & (labels_b >= 0)
    if valid.sum() < 2:
        return float("nan")
    a = labels_a[valid]
    b = labels_b[valid]
    unique_a = sorted(set(int(v) for v in a.tolist()))
    unique_b = sorted(set(int(v) for v in b.tolist()))
    matrix = np.zeros((len(unique_a), len(unique_b)), dtype=np.float32)
    for i, ca in enumerate(unique_a):
        for j, cb in enumerate(unique_b):
            matrix[i, j] = float(np.sum((a == ca) & (b == cb)))
    if not matrix.size:
        return float("nan")
    row_ind, col_ind = linear_sum_assignment(-matrix)
    matched = matrix[row_ind, col_ind].sum()
    return float(matched / max(1, len(a)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate perception modelling choices across representations and projections")
    parser.add_argument("--documents", type=Path, default=data_dir() / "documents.parquet")
    parser.add_argument("--sentence-embeddings", type=Path, default=data_dir() / "embeddings_sentence_transformer.npy")
    parser.add_argument("--tfidf", type=Path, default=data_dir() / "tfidf_matrix.npz")
    parser.add_argument("--representation-similarity", type=Path, default=data_dir() / "representation_similarity.csv")
    parser.add_argument("--model-runs", type=Path, default=data_dir() / "model_runs.csv")
    parser.add_argument("--assignments", type=Path, default=data_dir() / "model_assignments.parquet")
    parser.add_argument("--reduction-validation", type=Path, default=data_dir() / "reduction_validation.csv")
    args = parser.parse_args()

    documents = load_documents(args.documents)
    doc_ids = documents["id"].astype(str).tolist()
    sentence_embeddings = load_matrix(args.sentence_embeddings)
    tfidf_matrix = load_tfidf(args.tfidf)
    representations = prepare_representation_matrices(sentence_embeddings, tfidf_matrix)

    assignment_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    run_store: dict[tuple[str, str, str, int], np.ndarray] = {}
    conf_store: dict[tuple[str, str, str, int], np.ndarray] = {}

    for representation_name, matrix in representations.items():
        if issparse(matrix):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=np.float32)
        for algorithm, parameters, seeds in parameter_grid():
            for seed in seeds:
                labels, confidence = run_single_clustering(representation_name, algorithm, matrix, parameters, seed)
                key = (representation_name, algorithm, params_key(parameters), seed)
                run_store[key] = labels
                conf_store[key] = confidence
                run_rows.append(
                    {
                        "representation": representation_name,
                        "algorithm": algorithm,
                        "parameters": json.dumps(parameters, sort_keys=True, ensure_ascii=True),
                        "seed": seed,
                        "cluster_count": cluster_count(labels),
                        "silhouette_score": safe_silhouette(matrix, labels),
                    }
                )
                for doc_id, label, conf in zip(doc_ids, labels, confidence):
                    assignment_rows.append(
                        {
                            "run_id": f"{representation_name}|{algorithm}|{params_key(parameters)}|{seed}",
                            "representation": representation_name,
                            "algorithm": algorithm,
                            "parameters": json.dumps(parameters, sort_keys=True, ensure_ascii=True),
                            "seed": seed,
                            "document_id": doc_id,
                            "cluster_id": int(label),
                            "confidence": float(conf),
                        }
                    )

    run_frame = pd.DataFrame(run_rows)
    stability_scores = []
    for (representation_name, algorithm, parameters_json, seed), labels in run_store.items():
        peers = [peer for peer_key, peer in run_store.items() if peer_key[:3] == (representation_name, algorithm, parameters_json) and peer_key[3] != seed]
        if peers:
            peer_scores = [adjusted_rand_score(labels, peer) for peer in peers]
            stability = float(np.mean(peer_scores))
        else:
            stability = float("nan")
        stability_scores.append({"representation": representation_name, "algorithm": algorithm, "parameters": parameters_json, "seed": seed, "stability_score": stability})

    run_frame = run_frame.merge(pd.DataFrame(stability_scores), on=["representation", "algorithm", "parameters", "seed"], how="left")
    args.model_runs.parent.mkdir(parents=True, exist_ok=True)
    run_frame.to_csv(args.model_runs, index=False)
    pd.DataFrame(assignment_rows).to_parquet(args.assignments, index=False)

    similarity_rows = []
    for algorithm in ("kmeans", "gmm", "spectral", "hdbscan"):
        for parameters_json in sorted({key[2] for key in run_store.keys() if key[1] == algorithm}):
            for seed in sorted({key[3] for key in run_store.keys() if key[1] == algorithm and key[2] == parameters_json}):
                key_a = ("sentence_transformer", algorithm, parameters_json, seed)
                key_b = ("tfidf", algorithm, parameters_json, seed)
                if key_a not in run_store or key_b not in run_store:
                    continue
                labels_a = run_store[key_a]
                labels_b = run_store[key_b]
                similarity_rows.append(
                    {
                        "algorithm": algorithm,
                        "parameters": parameters_json,
                        "seed": seed,
                        "representation_a": "sentence_transformer",
                        "representation_b": "tfidf",
                        "adjusted_rand_index": float(adjusted_rand_score(labels_a, labels_b)),
                        "normalized_mutual_info": float(normalized_mutual_info_score(labels_a, labels_b)),
                        "best_cluster_overlap": best_label_overlap(labels_a, labels_b),
                    }
                )

    pd.DataFrame(similarity_rows).to_csv(args.representation_similarity, index=False)

    reduction_rows = []
    for representation_name, matrix in representations.items():
        if issparse(matrix):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=np.float32)
        reduction_seed = 42
        base_k = 5
        base_labels, _ = run_kmeans(reduce_for_clustering(matrix), base_k, reduction_seed)
        for method in ("umap", "pca", "tsne"):
            coords = reduction_projection(matrix, method, reduction_seed)
            for k in (3, 5, 7):
                projected_labels, _ = run_kmeans(coords, min(k, max(2, len(matrix) - 1)), reduction_seed)
                reduction_rows.append(
                    {
                        "representation": representation_name,
                        "method": method,
                        "k": k,
                        "neighbourhood_preservation": float(trustworthiness(matrix, coords, n_neighbors=min(10, max(2, len(matrix) // 10 or 2)))),
                        "cluster_consistency_ari": float(adjusted_rand_score(base_labels, projected_labels)),
                    }
                )

    pd.DataFrame(reduction_rows).to_csv(args.reduction_validation, index=False)
    print(f"Wrote model runs to {args.model_runs}")
    print(f"Wrote representation similarity to {args.representation_similarity}")
    print(f"Wrote reduction validation to {args.reduction_validation}")


if __name__ == "__main__":
    main()

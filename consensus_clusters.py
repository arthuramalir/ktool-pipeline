from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from strapi_utils import data_dir


def load_assignments(assignments_path: Path, fallback_clusters_path: Path) -> pd.DataFrame:
    if assignments_path.exists():
        return pd.read_parquet(assignments_path)
    clusters = pd.read_parquet(fallback_clusters_path)
    rows = []
    for _, row in clusters.iterrows():
        rows.append(
            {
                "run_id": f"{row['algorithm']}|{row['algorithm']}|legacy|0",
                "representation": "sentence_transformer",
                "algorithm": row["algorithm"],
                "parameters": "legacy",
                "seed": 0,
                "document_id": row["document_id"],
                "cluster_id": int(row["cluster_id"]),
                "confidence": float(row["probability"]),
            }
        )
    return pd.DataFrame(rows)


def entropy_from_distribution(values: np.ndarray) -> float:
    values = values.astype(np.float64)
    total = values.sum()
    if total <= 0:
        return 0.0
    probs = values / total
    probs = probs[probs > 0]
    if not len(probs):
        return 0.0
    entropy = float(-(probs * np.log(probs)).sum())
    normalizer = np.log(len(values)) if len(values) > 1 else 1.0
    return entropy / normalizer if normalizer else entropy


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute consensus agreement across clustering algorithms")
    parser.add_argument("--clusters", type=Path, default=data_dir() / "cluster_results.parquet")
    parser.add_argument("--assignments", type=Path, default=data_dir() / "model_assignments.parquet")
    parser.add_argument("--documents", type=Path, default=data_dir() / "documents.parquet")
    parser.add_argument("--consensus", type=Path, default=data_dir() / "consensus_matrix.npy")
    parser.add_argument("--stability", type=Path, default=data_dir() / "cluster_stability.csv")
    parser.add_argument("--output", type=Path, default=data_dir() / "consensus_perception_space.parquet")
    args = parser.parse_args()

    assignments = load_assignments(args.assignments, args.clusters)
    documents = pd.read_parquet(args.documents)
    doc_ids = documents["id"].astype(str).tolist()
    doc_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    consensus = np.zeros((len(doc_ids), len(doc_ids)), dtype=np.float32)
    counts = np.zeros((len(doc_ids), len(doc_ids)), dtype=np.float32)

    for run_id, run in assignments.groupby("run_id"):
        cluster_to_members = run.groupby("cluster_id")["document_id"].apply(list)
        for members in cluster_to_members:
            member_indices = [doc_index[d] for d in members if d in doc_index and int(run.set_index("document_id").loc[d, "cluster_id"]) >= 0]
            for i in member_indices:
                for j in member_indices:
                    consensus[i, j] += 1.0
            for i in member_indices:
                for j in member_indices:
                    counts[i, j] += 1.0

    counts[counts == 0] = 1.0
    consensus = consensus / counts
    np.save(args.consensus, consensus)

    threshold = 0.6
    adjacency = (consensus >= threshold).astype(np.int32)
    np.fill_diagonal(adjacency, 1)
    n_components, labels = connected_components(adjacency, directed=False, connection="weak")

    component_members = {component_id: np.where(labels == component_id)[0].tolist() for component_id in range(n_components)}
    cluster_rows = []
    for component_id, members in component_members.items():
        if len(members) == 0:
            continue
        block = consensus[np.ix_(members, members)]
        block_values = block[np.triu_indices_from(block, k=1)]
        cluster_rows.append(
            {
                "cluster_consensus_id": int(component_id),
                "cluster_confidence": float(np.nanmean(block_values) if len(block_values) else 1.0),
                "size": int(len(members)),
                "mean_pairwise_agreement": float(np.nanmean(block_values) if len(block_values) else 1.0),
            }
        )

    doc_rows = []
    for doc_id in doc_ids:
        idx = doc_index[doc_id]
        cluster_scores = []
        for component_id, members in component_members.items():
            if not members:
                continue
            score = float(np.mean(consensus[idx, members]))
            cluster_scores.append((component_id, score))
        if not cluster_scores:
            cluster_scores = [(0, 1.0)]
        cluster_scores.sort(key=lambda item: item[1], reverse=True)
        weights = np.asarray([score for _, score in cluster_scores], dtype=np.float32)
        entropy_score = entropy_from_distribution(weights)
        max_score = float(weights.max()) if len(weights) else 0.0
        norm = float(weights.sum()) if float(weights.sum()) > 0 else 1.0
        doc_rows.append(
            {
                "document_id": doc_id,
                "cluster_consensus_id": int(cluster_scores[0][0]),
                "cluster_confidence": float(cluster_scores[0][1]),
                "entropy_score": float(entropy_score),
                "ambiguity_score": float(1.0 - (max_score / norm if norm else 0.0)),
            }
        )

    stability_rows = pd.DataFrame(cluster_rows)
    if not stability_rows.empty:
        stability_rows = stability_rows.rename(columns={"cluster_confidence": "stability"})
        stability_rows.to_csv(args.stability, index=False)
    else:
        pd.DataFrame(columns=["cluster_consensus_id", "stability", "size", "mean_pairwise_agreement"]).to_csv(args.stability, index=False)

    pd.DataFrame(doc_rows).to_parquet(args.output, index=False)
    print(f"Wrote consensus matrix to {args.consensus}")
    print(f"Wrote cluster stability summary to {args.stability}")
    print(f"Wrote consensus perception space to {args.output}")


if __name__ == "__main__":
    main()

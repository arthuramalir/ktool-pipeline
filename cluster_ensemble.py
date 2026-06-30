from __future__ import annotations

import argparse
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

from strapi_utils import data_dir


def load_documents(documents_path: Path) -> pd.DataFrame:
    return pd.read_parquet(documents_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a clustering ensemble over document embeddings")
    parser.add_argument("--documents", type=Path, default=data_dir() / "documents.parquet")
    parser.add_argument("--sentence-embeddings", type=Path, default=data_dir() / "embeddings_sentence_transformer.npy")
    parser.add_argument("--tfidf", type=Path, default=data_dir() / "tfidf_matrix.npz")
    parser.add_argument("--output", type=Path, default=data_dir() / "cluster_results.parquet")
    args = parser.parse_args()

    documents = load_documents(args.documents)
    doc_ids = documents["id"].astype(str).tolist()
    sentence_embeddings = np.load(args.sentence_embeddings)
    tfidf = sparse.load_npz(args.tfidf)
    tfidf_dense = tfidf.toarray()

    rows = []
    for k in range(3, 9):
        labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(sentence_embeddings)
        for doc_id, label in zip(doc_ids, labels):
            rows.append({"document_id": doc_id, "algorithm": f"kmeans_k{k}", "cluster_id": int(label), "probability": 1.0})

    hdb = hdbscan.HDBSCAN(min_cluster_size=max(5, len(doc_ids) // 50 or 5), prediction_data=True)
    hdb_labels = hdb.fit_predict(sentence_embeddings)
    hdb_conf = getattr(hdb, "probabilities_", np.ones(len(doc_ids)))
    for doc_id, label, conf in zip(doc_ids, hdb_labels, hdb_conf):
        rows.append({"document_id": doc_id, "algorithm": "hdbscan", "cluster_id": int(label), "probability": float(conf)})

    max_clusters = min(8, max(2, len(doc_ids) // 10 or 2))
    gmm = GaussianMixture(n_components=max_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(sentence_embeddings)
    gmm_probs = gmm.predict_proba(sentence_embeddings).max(axis=1)
    for doc_id, label, conf in zip(doc_ids, gmm_labels, gmm_probs):
        rows.append({"document_id": doc_id, "algorithm": "gaussian_mixture", "cluster_id": int(label), "probability": float(conf)})

    spectral = SpectralClustering(n_clusters=min(6, max(2, len(doc_ids) // 10 or 2)), random_state=42, affinity="nearest_neighbors")
    spectral_labels = spectral.fit_predict(sentence_embeddings)
    for doc_id, label in zip(doc_ids, spectral_labels):
        rows.append({"document_id": doc_id, "algorithm": "spectral", "cluster_id": int(label), "probability": 1.0})

    frame = pd.DataFrame(rows)
    frame.to_parquet(args.output, index=False)
    print(f"Wrote cluster ensemble results to {args.output}")


if __name__ == "__main__":
    main()

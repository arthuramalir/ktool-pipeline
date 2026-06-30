from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from strapi_utils import data_dir, project_root


def load_documents(documents_path: Path) -> pd.DataFrame:
    return pd.read_parquet(documents_path)


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)


def embed(method: str, matrix: np.ndarray) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=42).fit_transform(matrix)
    if method == "tsne":
        perplexity = max(5, min(30, len(matrix) // 3 if len(matrix) > 9 else 5))
        return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(matrix)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, max(3, len(matrix) // 10 or 3)))
    return reducer.fit_transform(matrix)


def save_figure(coords: pd.DataFrame, method: str, figure_path: Path) -> None:
    subset = coords[coords["method"] == method]
    plt.figure(figsize=(10, 8))
    plt.scatter(subset["x"], subset["y"], s=16, alpha=0.7)
    plt.title(f"Perception space: {method.upper()}")
    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Project documents into multiple perception spaces")
    parser.add_argument("--documents", type=Path, default=data_dir() / "documents.parquet")
    parser.add_argument("--sentence-embeddings", type=Path, default=data_dir() / "embeddings_sentence_transformer.npy")
    parser.add_argument("--tfidf", type=Path, default=data_dir() / "tfidf_matrix.npz")
    parser.add_argument("--output", type=Path, default=data_dir() / "perception_coordinates.parquet")
    parser.add_argument("--figures", type=Path, default=project_root() / "figures")
    args = parser.parse_args()

    documents = load_documents(args.documents)
    doc_ids = documents["id"].astype(str).tolist()
    sentence_embeddings = load_embeddings(args.sentence_embeddings)
    tfidf_matrix = sparse.load_npz(args.tfidf)

    coordinates = []
    for embedding_type, matrix in [("sentence_transformer", sentence_embeddings), ("tfidf", tfidf_matrix.toarray())]:
        for method in ("umap", "pca", "tsne"):
            points = embed(method, matrix)
            for doc_id, (x, y) in zip(doc_ids, points):
                coordinates.append(
                    {
                        "id": doc_id,
                        "x": float(x),
                        "y": float(y),
                        "method": method,
                        "embedding_type": embedding_type,
                    }
                )

    frame = pd.DataFrame(coordinates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)

    save_figure(frame[frame["embedding_type"] == "sentence_transformer"], "umap", args.figures / "perception_space_umap.png")
    save_figure(frame[frame["embedding_type"] == "sentence_transformer"], "pca", args.figures / "perception_space_pca.png")
    save_figure(frame[frame["embedding_type"] == "sentence_transformer"], "tsne", args.figures / "perception_space_tsne.png")
    print(f"Wrote perception coordinates to {args.output}")


if __name__ == "__main__":
    main()

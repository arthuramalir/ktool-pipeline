from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from strapi_utils import data_dir, project_root


def load_perception_space(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"id", "x", "y", "method", "embedding_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Perception space is missing required columns: {sorted(missing)}")
    return frame


def load_documents(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"id", "entity_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Documents are missing required columns: {sorted(missing)}")
    return frame


def build_graph_communities(graph: nx.Graph) -> tuple[dict[str, int], dict[str, float]]:
    simple_graph = nx.Graph(graph)
    communities = list(nx.algorithms.community.louvain_communities(simple_graph, seed=42))
    membership = {str(node): idx for idx, community in enumerate(communities) for node in community}
    degree = nx.degree_centrality(simple_graph)
    return membership, {str(node): float(score) for node, score in degree.items()}


def make_overlap_frame(documents: pd.DataFrame, coords: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
    space = coords[(coords["embedding_type"] == "sentence_transformer") & (coords["method"] == "umap")][["id", "x", "y"]].copy()
    merged = documents[["id", "entity_type"]].merge(space, on="id", how="inner")
    membership, degree = build_graph_communities(graph)
    merged["graph_community"] = merged["id"].map(membership).fillna(-1).astype(int)
    merged["degree_centrality"] = merged["id"].map(degree).fillna(0.0).astype(float)
    return merged


def perception_summary(overlap: pd.DataFrame) -> pd.DataFrame:
    perceptions = overlap[overlap["entity_type"].astype(str) == "perception"].copy()
    if perceptions.empty:
        return pd.DataFrame(columns=["graph_community", "count", "mean_degree", "mean_x", "mean_y"])
    return (
        perceptions.groupby("graph_community")
        .agg(count=("id", "count"), mean_degree=("degree_centrality", "mean"), mean_x=("x", "mean"), mean_y=("y", "mean"))
        .reset_index()
        .sort_values(["count", "graph_community"], ascending=[False, True])
    )


def save_overlap_plot(overlap: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    perceptions = overlap[overlap["entity_type"].astype(str) == "perception"].copy()
    others = overlap[overlap["entity_type"].astype(str) != "perception"].copy()

    plt.figure(figsize=(10, 8))
    if not others.empty:
        scatter = plt.scatter(others["x"], others["y"], c=others["graph_community"], s=14, alpha=0.35, cmap="tab20", linewidths=0)
        plt.colorbar(scatter, label="Graph community")
    if not perceptions.empty:
        plt.scatter(perceptions["x"], perceptions["y"], c="black", s=120, marker="*", edgecolors="white", linewidths=0.7, label="Perception nodes")
        for _, row in perceptions.iterrows():
            plt.annotate(str(row["id"]), (row["x"], row["y"]), xytext=(4, 4), textcoords="offset points", fontsize=8)

    plt.title("Perception-space overlap with graph communities")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    if not perceptions.empty:
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Join perception-space coordinates with graph communities")
    parser.add_argument("--documents", type=Path, default=data_dir() / "documents.parquet")
    parser.add_argument("--perception-space", type=Path, default=data_dir() / "perception_coordinates.parquet")
    parser.add_argument("--graph", type=Path, default=data_dir() / "graph.graphml")
    parser.add_argument("--output", type=Path, default=data_dir() / "perception_graph_overlap.parquet")
    parser.add_argument("--summary-output", type=Path, default=data_dir() / "perception_graph_overlap_summary.parquet")
    parser.add_argument("--figure", type=Path, default=project_root() / "figures" / "perception_graph_overlap_umap.png")
    args = parser.parse_args()

    documents = load_documents(args.documents)
    coords = load_perception_space(args.perception_space)
    graph = nx.read_graphml(args.graph).to_undirected()

    overlap = make_overlap_frame(documents, coords, graph)
    summary = perception_summary(overlap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overlap.to_parquet(args.output, index=False)
    summary.to_parquet(args.summary_output, index=False)
    save_overlap_plot(overlap, args.figure)

    print(f"Wrote overlap table to {args.output}")
    print(f"Wrote perception summary to {args.summary_output}")
    print(f"Wrote overlap figure to {args.figure}")


if __name__ == "__main__":
    main()
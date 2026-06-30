from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from strapi_utils import data_dir, project_root


def load_documents(documents_path: Path) -> pd.DataFrame:
    return pd.read_parquet(documents_path)


def load_perception_coordinates(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_consensus_map(documents: pd.DataFrame, consensus: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    umap_coords = coords[(coords["method"] == "umap") & (coords["embedding_type"] == "sentence_transformer")][["id", "x", "y"]]
    docs = documents.merge(umap_coords, on="id", how="inner")
    if not consensus.empty:
        docs = docs.merge(consensus, left_on="id", right_on="document_id", how="left")
    else:
        docs["cluster_consensus_id"] = -1
        docs["cluster_confidence"] = 0.0
        docs["entropy_score"] = 0.0
        docs["ambiguity_score"] = 0.0
    docs["cluster_consensus_id"] = docs["cluster_consensus_id"].fillna(-1).astype(int)
    docs["cluster_confidence"] = docs["cluster_confidence"].fillna(0.0).astype(float)
    docs["entropy_score"] = docs["entropy_score"].fillna(0.0).astype(float)
    docs["ambiguity_score"] = docs["ambiguity_score"].fillna(0.0).astype(float)
    docs["text_excerpt"] = docs["text"].fillna("").astype(str).str.slice(0, 220)
    return docs


def write_html(fig: go.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def consensus_landscape(consensus_map: pd.DataFrame, output_path: Path) -> None:
    size_scale = np.clip((consensus_map["ambiguity_score"].fillna(0.0) * 35) + 8, 6, 42)
    opacity = np.clip(consensus_map["cluster_confidence"].fillna(0.0), 0.15, 1.0)
    fig = go.Figure(
        data=
            go.Scattergl(
                x=consensus_map["x"],
                y=consensus_map["y"],
                mode="markers",
                marker=dict(
                    size=size_scale,
                    color=consensus_map["cluster_consensus_id"],
                    colorscale="Viridis",
                    opacity=opacity,
                    showscale=True,
                    colorbar=dict(title="Consensus cluster"),
                    line=dict(width=0.3, color="rgba(255,255,255,0.5)"),
                ),
                customdata=np.stack(
                    [
                        consensus_map["id"],
                        consensus_map["entity_type"],
                        consensus_map["text_excerpt"],
                        consensus_map["cluster_confidence"],
                        consensus_map["ambiguity_score"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Entity type: %{customdata[1]}<br>"
                    "Cluster confidence: %{customdata[3]:.2f}<br>"
                    "Uncertainty: %{customdata[4]:.2f}<br>"
                    "Text: %{customdata[2]}<extra></extra>"
                ),
            )
    )
    fig.update_layout(
        title="Consensus Perception Landscape",
        xaxis_title="UMAP x",
        yaxis_title="UMAP y",
        template="plotly_white",
        width=1200,
        height=860,
    )
    write_html(fig, output_path)


def co_clustering_matrix(assignments: pd.DataFrame, document_ids: list[str]) -> np.ndarray:
    index = {doc_id: idx for idx, doc_id in enumerate(document_ids)}
    matrix = np.zeros((len(document_ids), len(document_ids)), dtype=np.float32)
    counts = np.zeros((len(document_ids), len(document_ids)), dtype=np.float32)
    for algorithm, subset in assignments.groupby("algorithm"):
        for cluster_id, group in subset.groupby("cluster_id"):
            members = [index[doc_id] for doc_id in group["document_id"].astype(str).tolist() if doc_id in index and int(cluster_id) >= 0]
            if len(members) < 2:
                continue
            matrix[np.ix_(members, members)] += 1.0
            counts[np.ix_(members, members)] += 1.0
    counts[counts == 0] = 1.0
    return matrix / counts


def cluster_overlap_network(assignments: pd.DataFrame, output_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_members: dict[str, set[str]] = {}
    for algorithm, subset in assignments.groupby("algorithm"):
        for cluster_id, group in subset.groupby("cluster_id"):
            if int(cluster_id) < 0:
                continue
            cluster_key = f"{algorithm.upper()}_Cluster_{int(cluster_id)}"
            cluster_members[cluster_key] = set(group["document_id"].astype(str).tolist())

    graph = nx.Graph()
    for node, members in cluster_members.items():
        graph.add_node(node, size=len(members), algorithm=node.split("_Cluster_")[0])

    nodes = list(cluster_members.items())
    for (left_name, left_members), (right_name, right_members) in combinations(nodes, 2):
        if left_name.split("_Cluster_")[0] == right_name.split("_Cluster_")[0]:
            continue
        union = left_members | right_members
        if not union:
            continue
        jaccard = len(left_members & right_members) / len(union)
        if jaccard <= 0:
            continue
        graph.add_edge(left_name, right_name, weight=float(jaccard))

    pos = nx.spring_layout(graph, seed=42, weight="weight") if graph.number_of_nodes() else {}
    edge_trace = []
    for left, right, data in graph.edges(data=True):
        x0, y0 = pos[left]
        x1, y1 = pos[right]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=max(0.5, data.get("weight", 0.05) * 6), color="rgba(90,90,90,0.45)"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    algorithm_palette = {"KMEANS": 1, "HDBSCAN": 2, "GMM": 3, "SPECTRAL": 4}
    for node, attrs in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(max(10, attrs.get("size", 1) * 2.5))
        node_color.append(algorithm_palette.get(attrs.get("algorithm", ""), 0))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        marker=dict(size=node_size, color=node_color, colorscale="Plasma", showscale=True, line=dict(width=0.5, color="white")),
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )

    fig = go.Figure(data=edge_trace + [node_trace])
    fig.update_layout(
        title="Cluster Overlap Network",
        template="plotly_white",
        width=1300,
        height=900,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    write_html(fig, output_path)

    edge_rows = [{"source": left, "target": right, "jaccard_similarity": data["weight"]} for left, right, data in graph.edges(data=True)]
    node_rows = [
        {"cluster_node": node, "algorithm": attrs.get("algorithm"), "size": attrs.get("size", 0)}
        for node, attrs in graph.nodes(data=True)
    ]
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def agreement_heatmap(assignments: pd.DataFrame, document_ids: list[str], output_path: Path) -> np.ndarray:
    matrix = co_clustering_matrix(assignments, document_ids)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=document_ids,
            y=document_ids,
            colorscale="Viridis",
            colorbar=dict(title="Co-cluster frequency"),
        )
    )
    fig.update_layout(
        title="Document Co-Clustering Similarity",
        template="plotly_white",
        width=1200,
        height=1000,
    )
    write_html(fig, output_path)
    return matrix


def noise_map(consensus_map: pd.DataFrame, assignments: pd.DataFrame, output_path: Path) -> None:
    hdbscan_noise = set(
        assignments[(assignments["algorithm"].str.lower() == "hdbscan") & (assignments["cluster_id"] < 0)]["document_id"].astype(str).tolist()
    )
    consensus_map = consensus_map.copy()
    consensus_map["noise_flag"] = consensus_map["id"].isin(hdbscan_noise)
    consensus_map["region"] = pd.cut(
        consensus_map["entropy_score"].fillna(0.0),
        bins=[-0.01, 0.35, 0.7, 1.01],
        labels=["stable perception regions", "contested perception regions", "unexplained areas"],
    )
    consensus_map["size"] = np.clip((consensus_map["ambiguity_score"].fillna(0.0) * 26) + 6, 5, 40)
    fig = go.Figure()
    for region, subset in consensus_map.groupby("region", dropna=False):
        fig.add_trace(
            go.Scattergl(
                x=subset["x"],
                y=subset["y"],
                mode="markers",
                name=str(region),
                marker=dict(
                    size=subset["size"],
                    color=np.where(subset["noise_flag"], 1, 0),
                    colorscale=[[0, "rgba(50,120,220,0.45)"], [1, "rgba(230,80,60,0.85)"]],
                    opacity=np.clip(subset["cluster_confidence"].fillna(0.0), 0.2, 1.0),
                    showscale=False,
                    line=dict(width=0.3, color="rgba(255,255,255,0.4)"),
                ),
                customdata=np.stack(
                    [subset["id"], subset["entity_type"], subset["text_excerpt"], subset["cluster_confidence"], subset["entropy_score"], subset["ambiguity_score"], subset["cluster_consensus_id"]],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Entity type: %{customdata[1]}<br>"
                    "Consensus cluster: %{customdata[6]}<br>"
                    "Confidence: %{customdata[3]:.2f}<br>"
                    "Entropy: %{customdata[4]:.2f}<br>"
                    "Ambiguity: %{customdata[5]:.2f}<br>"
                    "Text: %{customdata[2]}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Perception Noise Map",
        template="plotly_white",
        width=1200,
        height=880,
        xaxis_title="UMAP x",
        yaxis_title="UMAP y",
        legend_title="Perception region",
    )
    write_html(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create perception uncertainty visualizations")
    parser.add_argument("--documents", type=Path, default=data_dir() / "documents.parquet")
    parser.add_argument("--perception-space", type=Path, default=data_dir() / "perception_coordinates.parquet")
    parser.add_argument("--consensus", type=Path, default=data_dir() / "consensus_perception_space.parquet")
    parser.add_argument("--clusters", type=Path, default=data_dir() / "cluster_results.parquet")
    parser.add_argument("--assignments", type=Path, default=data_dir() / "model_assignments.parquet")
    parser.add_argument("--figures", type=Path, default=project_root() / "figures")
    args = parser.parse_args()

    documents = load_documents(args.documents)
    coords = load_perception_coordinates(args.perception_space)
    consensus = pd.read_parquet(args.consensus) if args.consensus.exists() else pd.DataFrame()
    clusters = pd.read_parquet(args.clusters) if args.clusters.exists() else pd.DataFrame()
    assignments = pd.read_parquet(args.assignments) if args.assignments.exists() else clusters.rename(columns={"probability": "confidence"})

    consensus_map = build_consensus_map(documents, consensus, coords)
    consensus_landscape(consensus_map, args.figures / "consensus_perception_landscape.html")

    if not assignments.empty:
        document_ids = documents["id"].astype(str).tolist()
        agreement_heatmap(assignments, document_ids, args.figures / "cluster_agreement_heatmap.html")
        cluster_overlap_network(assignments, args.figures / "cluster_overlap_network.html")
        noise_map(consensus_map, assignments, args.figures / "perception_noise_map.html")
    else:
        empty_heatmap = go.Figure()
        write_html(empty_heatmap, args.figures / "cluster_agreement_heatmap.html")
        write_html(go.Figure(), args.figures / "cluster_overlap_network.html")
        write_html(go.Figure(), args.figures / "perception_noise_map.html")

    print(f"Wrote consensus perception landscape to {args.figures / 'consensus_perception_landscape.html'}")
    print(f"Wrote cluster agreement heatmap to {args.figures / 'cluster_agreement_heatmap.html'}")
    print(f"Wrote cluster overlap network to {args.figures / 'cluster_overlap_network.html'}")
    print(f"Wrote perception noise map to {args.figures / 'perception_noise_map.html'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
DATA_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR
ANALYTICS_DIR = DATA_DIR / "analytics"
ANALYSIS_DIR = DATA_DIR / "analysis"

INITIATIVE_TYPES = {"project", "pilot", "prototype"}
#NOT NECESSARY
"""
NODE_COLUMNS = [
    "global_id",
    "native_id",
    "node_type",
    "label",
    "description",
    "methodological_phase",
    "platform_id",
    "source_table",
]
EDGE_COLUMNS = [
    "edge_id",
    "source_global_id",
    "target_global_id",
    "edge_type",
    "edge_family",
    "methodological_phase",
    "directed",
    "weight",
    "connection_type",
    "evidence_source",
    "platform_id",
]
"""

def ensure_output_dirs() -> None:
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_table(filename: str, *subfolders: str) -> pd.DataFrame:
    candidates = [DATA_DIR / filename]
    candidates.extend(DATA_DIR / folder / filename for folder in subfolders)
    for i, path in enumerate(candidates):
        frame = read_csv_safe(path)
        if not frame.empty:
            return frame
        if path.exists() and i == len(candidates) - 1:
            return frame
    return pd.DataFrame()


def load_nodes_edges() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = read_csv_safe(ANALYTICS_DIR / "nodes.csv")
    edges = read_csv_safe(ANALYTICS_DIR / "edges.csv")
    if nodes.empty:
        nodes = read_csv_safe(DATA_DIR / "nodes.csv")
    if edges.empty:
        edges = read_csv_safe(DATA_DIR / "edges.csv")
    return nodes, edges


def write_json(path: Path, data: dict | list) -> None:
    ensure_output_dirs()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_frame(df: pd.DataFrame, filename: str) -> Path:
    ensure_output_dirs()
    path = ANALYSIS_DIR / filename
    df.to_csv(path, index=False)
    return path


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def build_graph(
    edge_families: Iterable[str] | None = None,
    node_types: Iterable[str] | None = None,
    directed: bool = False,
) -> nx.Graph | nx.MultiDiGraph:
    nodes, edges = load_nodes_edges()
    edge_family_set = set(edge_families) if edge_families else None
    node_type_set = set(node_types) if node_types else None

    if node_type_set is not None and not nodes.empty:
        nodes = nodes[nodes["node_type"].isin(node_type_set)].copy()
    valid_nodes = set(nodes["global_id"].astype(str)) if not nodes.empty else set()

    if edge_family_set is not None and not edges.empty:
        edges = edges[edges["edge_family"].isin(edge_family_set)].copy()
    if valid_nodes and not edges.empty:
        edges = edges[
            edges["source_global_id"].astype(str).isin(valid_nodes)
            & edges["target_global_id"].astype(str).isin(valid_nodes)
        ].copy()

    graph = nx.MultiDiGraph() if directed else nx.Graph()
    for _, row in nodes.iterrows():
        node_id = str(row["global_id"])
        attrs = {col: "" if pd.isna(row[col]) else row[col] for col in nodes.columns if col != "global_id"}
        graph.add_node(node_id, **attrs)

    for _, row in edges.iterrows():
        source = str(row["source_global_id"])
        target = str(row["target_global_id"])
        attrs = {col: "" if pd.isna(row[col]) else row[col] for col in edges.columns if col not in {"source_global_id", "target_global_id"}}
        graph.add_edge(source, target, **attrs)
    return graph


def simple_graph(graph: nx.Graph | nx.MultiDiGraph) -> nx.Graph:
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes(data=True))
    for source, target, attrs in graph.edges(data=True):
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] = float(undirected[source][target].get("weight", 1)) + 1.0
        else:
            clean_attrs = dict(attrs)
            clean_attrs["weight"] = 1.0
            undirected.add_edge(source, target, **clean_attrs)
    return undirected


def largest_component_subgraph(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph.copy()
    components = list(nx.connected_components(graph))
    if not components:
        return graph.copy()
    largest = max(components, key=len)
    return graph.subgraph(largest).copy()


def safe_centrality(graph: nx.Graph) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["global_id", "degree", "degree_centrality", "betweenness_centrality", "closeness_centrality"])
    degree = dict(graph.degree())
    degree_c = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, normalized=True) if graph.number_of_edges() else {node: 0.0 for node in graph.nodes}
    closeness = nx.closeness_centrality(graph) if graph.number_of_edges() else {node: 0.0 for node in graph.nodes}
    return pd.DataFrame(
        [
            {
                "global_id": node,
                "degree": degree.get(node, 0),
                "degree_centrality": degree_c.get(node, 0.0),
                "betweenness_centrality": betweenness.get(node, 0.0),
                "closeness_centrality": closeness.get(node, 0.0),
            }
            for node in graph.nodes
        ]
    )


def project_bipartite(source_type: str, target_type: str) -> nx.Graph:
    nodes, edges = load_nodes_edges()
    if nodes.empty or edges.empty:
        return nx.Graph()
    node_types = nodes.set_index("global_id")["node_type"].to_dict()
    bipartite_edges = []
    for _, row in edges.iterrows():
        source = str(row["source_global_id"])
        target = str(row["target_global_id"])
        source_node_type = node_types.get(source)
        target_node_type = node_types.get(target)
        if source_node_type == source_type and target_node_type == target_type:
            bipartite_edges.append((source, target))
        elif source_node_type == target_type and target_node_type == source_type:
            bipartite_edges.append((target, source))

    projection = nx.Graph()
    by_target: dict[str, set[str]] = {}
    for source, target in bipartite_edges:
        by_target.setdefault(target, set()).add(source)
        projection.add_node(source)
    for source_nodes in by_target.values():
        source_list = sorted(source_nodes)
        for i, left in enumerate(source_list):
            for right in source_list[i + 1 :]:
                if projection.has_edge(left, right):
                    projection[left][right]["weight"] += 1
                else:
                    projection.add_edge(left, right, weight=1)
    return projection


def enrich_with_node_metadata(frame: pd.DataFrame, id_col: str = "global_id") -> pd.DataFrame:
    nodes, _ = load_nodes_edges()
    if frame.empty or nodes.empty or id_col not in frame.columns:
        return frame
    metadata = nodes[["global_id", "label", "node_type", "methodological_phase"]].drop_duplicates()
    return frame.merge(metadata, left_on=id_col, right_on="global_id", how="left", suffixes=("", "_node"))

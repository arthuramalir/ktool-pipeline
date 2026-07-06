from __future__ import annotations

import hashlib
import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from graph_utils import ANALYSIS_DIR, DATA_DIR, ensure_output_dirs, load_nodes_edges


GNN_DIR = ANALYSIS_DIR / "gnn"
TEXT_HASH_DIM = 256
DEFAULT_ANON_TYPES = {"citizen", "person", "resident"}
SENSITIVE_CANDIDATES = [
    "gender",
    "sex",
    "age",
    "age_group",
    "ethnicity",
    "race",
    "disability",
    "income",
    "postcode",
    "location",
    "address",
    "nationality",
]

def safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def to_vector(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return np.asarray(parsed, dtype=np.float32)
        except Exception:
            return np.asarray([], dtype=np.float32)
    return np.asarray([], dtype=np.float32)


def load_embedding_map(candidates: Iterable[Path]) -> tuple[dict[str, np.ndarray], int, str]:
    for path in candidates:
        if not path.exists() or path.stat().st_size == 0:
            continue
        if path.suffix not in {".csv", ".parquet"}:
            continue
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        if "embedding" not in frame.columns:
            continue
        key_column = next((col for col in ["global_id", "entity_id", "id", "doc_id", "node_id"] if col in frame.columns), None)
        if key_column is None:
            continue
        mapping: dict[str, np.ndarray] = {}
        dim = 0
        for _, row in frame.iterrows():
            vector = to_vector(row["embedding"])
            if vector.size == 0:
                continue
            dim = int(vector.shape[0])
            mapping[safe_str(row[key_column])] = vector
        if mapping:
            return mapping, dim, str(path)
    return {}, 0, ""


def build_simple_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for _, row in nodes.iterrows():
        graph.add_node(safe_str(row["global_id"]))
    for _, row in edges.iterrows():
        source = safe_str(row.get("source_global_id"))
        target = safe_str(row.get("target_global_id"))
        if source and target and source != target:
            graph.add_edge(source, target)
    return graph


def simple_pagerank(graph: nx.Graph, alpha: float = 0.85, max_iter: int = 100, tol: float = 1.0e-8) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 1:
        return {nodes[0]: 1.0}
    rank = {node: 1.0 / n for node in nodes}
    out_degree = {node: graph.degree(node) for node in nodes}
    damping = (1.0 - alpha) / n
    for _ in range(max_iter):
        new_rank = {node: damping for node in nodes}
        dangling_mass = sum(rank[node] for node in nodes if out_degree[node] == 0)
        dangling_share = alpha * dangling_mass / n
        for node in nodes:
            new_rank[node] += dangling_share
        for node in nodes:
            degree = out_degree[node]
            if degree == 0:
                continue
            share = alpha * rank[node] / degree
            for neighbor in graph.neighbors(node):
                new_rank[neighbor] += share
        delta = sum(abs(new_rank[node] - rank[node]) for node in nodes)
        rank = new_rank
        if delta < tol:
            break
    total = sum(rank.values()) or 1.0
    return {node: float(value / total) for node, value in rank.items()}


def graph_metrics(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty:
        return pd.DataFrame()

    graph = build_simple_graph(nodes, edges)
    degree = dict(graph.degree()) if graph.number_of_nodes() else {}
    degree_centrality = nx.degree_centrality(graph) if graph.number_of_nodes() else {}
    betweenness = nx.betweenness_centrality(graph, normalized=True) if graph.number_of_edges() else {}
    closeness = nx.closeness_centrality(graph) if graph.number_of_edges() else {}
    pagerank = simple_pagerank(graph) if graph.number_of_edges() else {}
    clustering = nx.clustering(graph) if graph.number_of_edges() else {}
    core_number = nx.core_number(graph) if graph.number_of_edges() else {}
    articulation_points = set(nx.articulation_points(graph)) if graph.number_of_nodes() > 2 and graph.number_of_edges() else set()
    bridge_nodes: set[str] = set()
    if graph.number_of_edges():
        for left, right in nx.bridges(graph):
            bridge_nodes.add(str(left))
            bridge_nodes.add(str(right))

    component_map: dict[str, int] = {}
    component_size_map: dict[str, int] = {}
    for component_id, component in enumerate(nx.connected_components(graph) if graph.number_of_nodes() else []):
        members = list(component)
        for node_id in members:
            component_map[str(node_id)] = component_id
            component_size_map[str(node_id)] = len(members)

    rows: list[dict] = []
    for _, row in nodes.iterrows():
        node_id = safe_str(row["global_id"])
        rows.append(
            {
                "global_id": node_id,
                "degree": float(degree.get(node_id, 0)),
                "degree_centrality": float(degree_centrality.get(node_id, 0.0)),
                "betweenness_centrality": float(betweenness.get(node_id, 0.0)),
                "closeness_centrality": float(closeness.get(node_id, 0.0)),
                "pagerank": float(pagerank.get(node_id, 0.0)),
                "clustering_coefficient": float(clustering.get(node_id, 0.0)),
                "core_number": float(core_number.get(node_id, 0)),
                "component_id": int(component_map.get(node_id, -1)),
                "component_size": int(component_size_map.get(node_id, 1 if node_id else 0)),
                "articulation_flag": float(node_id in articulation_points),
                "bridge_incident_flag": float(node_id in bridge_nodes),
            }
        )
    return pd.DataFrame(rows)


def auto_sensitive_columns(nodes: pd.DataFrame) -> list[str]:
    matches = []
    for column in nodes.columns:
        lower = column.lower()
        if any(candidate in lower for candidate in SENSITIVE_CANDIDATES):
            matches.append(column)
    return matches


def build_sensitive_matrix(nodes: pd.DataFrame, sensitive_columns: list[str]) -> np.ndarray:
    if not sensitive_columns:
        return np.empty((len(nodes), 0), dtype=np.float32)
    frame = pd.DataFrame(index=nodes.index)
    for column in sensitive_columns:
        if column not in nodes.columns:
            continue
        series = nodes[column]
        if pd.api.types.is_numeric_dtype(series):
            frame[column] = pd.to_numeric(series, errors="coerce").fillna(0.0)
        else:
            dummies = pd.get_dummies(series.fillna("unknown").astype(str), prefix=f"sens_{column}")
            frame = pd.concat([frame, dummies], axis=1)
    if frame.empty:
        return np.empty((len(nodes), 0), dtype=np.float32)
    return frame.fillna(0.0).astype(np.float32).to_numpy()


def standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def hash_token(token: str, dimension: int) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % dimension


def orthogonalize_vectors(vectors: np.ndarray, sensitive_matrix: np.ndarray) -> np.ndarray:
    if vectors.size == 0 or sensitive_matrix.size == 0:
        return vectors
    centered = sensitive_matrix - sensitive_matrix.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0):
        return vectors
    projector = centered @ np.linalg.pinv(centered)
    return vectors - projector @ vectors


def build_text_matrix(nodes: pd.DataFrame, dimension: int = TEXT_HASH_DIM) -> np.ndarray:
    label = nodes.get("label", pd.Series(index=nodes.index, dtype=str)).fillna("").astype(str)
    description = nodes.get("description", pd.Series(index=nodes.index, dtype=str)).fillna("").astype(str)
    text = (label + " " + description).str.replace(r"\s+", " ", regex=True).str.strip().tolist()
    if not text:
        return np.empty((0, dimension), dtype=np.float32)
    matrix = np.zeros((len(text), dimension), dtype=np.float32)
    for row_index, entry in enumerate(text):
        tokens = re.findall(r"[A-Za-z0-9_]+", entry.lower())
        if not tokens:
            continue
        for token in tokens:
            matrix[row_index, hash_token(token, dimension)] += 1.0
        row_norm = float(np.linalg.norm(matrix[row_index]))
        if row_norm > 0:
            matrix[row_index] /= row_norm
    return matrix


def build_semantic_matrix(nodes: pd.DataFrame, embedding_map: dict[str, np.ndarray], embedding_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if not embedding_map or embedding_dim <= 0:
        return np.empty((len(nodes), 0), dtype=np.float32), np.zeros(len(nodes), dtype=bool)
    matrix = np.zeros((len(nodes), embedding_dim), dtype=np.float32)
    coverage = np.zeros(len(nodes), dtype=bool)
    for idx, row in nodes.iterrows():
        node_id = safe_str(row["global_id"])
        vector = embedding_map.get(node_id)
        if vector is None:
            native_id = safe_str(row.get("native_id"))
            vector = embedding_map.get(native_id)
        if vector is None or vector.shape[0] != embedding_dim:
            continue
        matrix[idx] = vector.astype(np.float32)
        coverage[idx] = True
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    matrix = matrix / row_norms
    return matrix, coverage


def provenance_score(row: pd.Series) -> float:
    family = safe_str(row.get("edge_family"))
    family_scores = {
        "declared_relational": 0.95,
        "listening": 0.85,
        "qualitative_narrative": 0.82,
        "interpretive": 0.74,
        "quote_semantic": 0.72,
        "narrative_claim": 0.78,
    }
    score = family_scores.get(family, 0.8)
    if bool(row.get("is_ai_generated")):
        score -= 0.08
    origin = safe_str(row.get("edge_origin")).lower()
    if origin == "source_data":
        score += 0.03
    elif origin in {"ai_inferred", "inference"}:
        score -= 0.05
    evidence = safe_str(row.get("evidence_source")).lower()
    if "semantic" in evidence:
        score -= 0.02
    return float(np.clip(score, 0.0, 1.0))


def confidence_components(src_vec: np.ndarray, tgt_vec: np.ndarray, row: pd.Series) -> tuple[float, float, float, float]:
    provenance = provenance_score(row)
    if src_vec.size == 0 or tgt_vec.size == 0:
        alignment = 0.5
        variance_score = 0.5
    else:
        src = src_vec.astype(np.float32)
        tgt = tgt_vec.astype(np.float32)
        src_norm = float(np.linalg.norm(src))
        tgt_norm = float(np.linalg.norm(tgt))
        if src_norm == 0.0 or tgt_norm == 0.0:
            alignment = 0.5
        else:
            alignment = float(np.dot(src, tgt) / (src_norm * tgt_norm))
            alignment = (alignment + 1.0) / 2.0
        alignment = float(np.clip(alignment, 0.0, 1.0))
        variance_value = float(np.var(np.vstack([src, tgt]), axis=0).mean())
        variance_score = float(1.0 / (1.0 + variance_value))
    omega = float(np.clip(0.45 * provenance + 0.35 * alignment + 0.20 * variance_score, 0.0, 1.0))
    return omega, provenance, alignment, variance_score


def add_edge_row(
    edge_rows: list[dict],
    source_idx: int,
    target_idx: int,
    source_id: str,
    target_id: str,
    relation_key: str,
    relation_id: int,
    row: pd.Series,
    direction: str,
    omega: float,
    provenance: float,
    alignment: float,
    variance_score: float,
    confidence_threshold: float,
) -> None:
    edge_rows.append(
        {
            "edge_id": safe_str(row.get("edge_id")) or f"{relation_key}:{source_id}->{target_id}:{direction}",
            "source_global_id": source_id,
            "target_global_id": target_id,
            "source_index": source_idx,
            "target_index": target_idx,
            "relation_key": relation_key,
            "relation_id": relation_id,
            "edge_type": safe_str(row.get("edge_type")),
            "edge_family": safe_str(row.get("edge_family")),
            "methodological_phase": safe_str(row.get("methodological_phase")),
            "directed": bool(row.get("directed", False)),
            "direction": direction,
            "evidence_source": safe_str(row.get("evidence_source")),
            "edge_origin": safe_str(row.get("edge_origin")),
            "generated_by": safe_str(row.get("generated_by")),
            "inference_method": safe_str(row.get("inference_method")),
            "platform_id": safe_str(row.get("platform_id")),
            "omega": omega,
            "provenance_score": provenance,
            "alignment_score": alignment,
            "alignment_variance_score": variance_score,
            "low_confidence": omega < confidence_threshold,
        }
    )


def build_gnn_outputs(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    semantic_embedding_map: dict[str, np.ndarray],
    semantic_embedding_dim: int,
    anonymize_types: set[str],
    sensitive_columns: list[str],
    text_hash_dim: int,
    confidence_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict, dict, np.ndarray]:
    nodes = nodes.copy().reset_index(drop=True)
    edges = edges.copy().reset_index(drop=True)

    nodes["global_id"] = nodes["global_id"].astype(str)
    nodes["node_type"] = nodes.get("node_type", pd.Series(dtype=str)).fillna("unknown").astype(str)
    nodes["label"] = nodes.get("label", pd.Series(dtype=str)).fillna("Unnamed").astype(str)
    nodes["description"] = nodes.get("description", pd.Series(dtype=str)).fillna("").astype(str)
    if "methodological_phase" not in nodes.columns:
        nodes["methodological_phase"] = "unknown"
    nodes["methodological_phase"] = nodes["methodological_phase"].fillna("unknown").astype(str)

    if anonymize_types:
        nodes["public_label"] = nodes["label"]
        nodes["is_anonymized"] = nodes["node_type"].str.lower().isin(anonymize_types)
        anonymized_nodes = nodes[nodes["is_anonymized"]].index.tolist()
        for anon_index, node_index in enumerate(anonymized_nodes, start=1):
            nodes.at[node_index, "public_label"] = f"Anon-{nodes.at[node_index, 'node_type']}-{anon_index}"
    else:
        nodes["public_label"] = nodes["label"]
        nodes["is_anonymized"] = False

    node_lookup = {node_id: idx for idx, node_id in enumerate(nodes["global_id"].tolist())}
    edge_mask = edges["source_global_id"].astype(str).isin(node_lookup) & edges["target_global_id"].astype(str).isin(node_lookup)
    edges = edges.loc[edge_mask].copy()
    edges["source_global_id"] = edges["source_global_id"].astype(str)
    edges["target_global_id"] = edges["target_global_id"].astype(str)

    metrics = graph_metrics(nodes, edges)
    nodes = nodes.merge(metrics, on="global_id", how="left")

    numeric_columns = [
        "degree",
        "degree_centrality",
        "betweenness_centrality",
        "closeness_centrality",
        "pagerank",
        "clustering_coefficient",
        "core_number",
        "component_id",
        "component_size",
        "articulation_flag",
        "bridge_incident_flag",
    ]
    numeric_frame = nodes[numeric_columns].fillna(0.0).astype(np.float32)
    if len(numeric_frame):
        numeric_scaled = standardize_matrix(numeric_frame.to_numpy(dtype=np.float32))
    else:
        numeric_scaled = np.empty((0, len(numeric_columns)), dtype=np.float32)

    type_frame = pd.get_dummies(nodes["node_type"].fillna("unknown").astype(str), prefix="type")
    phase_frame = pd.get_dummies(nodes["methodological_phase"].fillna("unknown").astype(str), prefix="phase")
    categorical_frame = pd.concat([type_frame, phase_frame], axis=1).reindex(nodes.index, fill_value=0)
    categorical_matrix = categorical_frame.astype(np.float32).to_numpy() if not categorical_frame.empty else np.empty((len(nodes), 0), dtype=np.float32)

    text_matrix = build_text_matrix(nodes, text_hash_dim)
    sensitive_columns = [column for column in sensitive_columns if column in nodes.columns]
    sensitive_matrix = build_sensitive_matrix(nodes, sensitive_columns)

    semantic_matrix, semantic_coverage = build_semantic_matrix(nodes, semantic_embedding_map, semantic_embedding_dim)
    alignment_source = semantic_matrix.copy() if semantic_matrix.size else text_matrix.copy()

    text_matrix = orthogonalize_vectors(text_matrix, sensitive_matrix)
    if semantic_matrix.size:
        semantic_matrix = orthogonalize_vectors(semantic_matrix, sensitive_matrix)

    feature_blocks = [block for block in [numeric_scaled, categorical_matrix, text_matrix, semantic_matrix] if block.size]
    x = np.concatenate(feature_blocks, axis=1) if feature_blocks else np.empty((len(nodes), 0), dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)

    node_feature_columns = [
        "global_id",
        "public_label",
        "label",
        "node_type",
        "methodological_phase",
        "is_anonymized",
    ] + numeric_columns + list(categorical_frame.columns)
    node_features = nodes[[col for col in node_feature_columns if col in nodes.columns]].copy()
    node_features["node_index"] = np.arange(len(nodes), dtype=int)
    node_features["semantic_embedding_covered"] = semantic_coverage
    node_features["sensitive_columns_used"] = ",".join(sensitive_columns)

    relation_map: dict[str, int] = {}
    edge_rows: list[dict] = []
    edge_index_pairs: list[tuple[int, int]] = []
    edge_types: list[int] = []
    edge_weights: list[float] = []
    edge_attrs: list[list[float]] = []

    for _, row in edges.iterrows():
        source_id = safe_str(row.get("source_global_id"))
        target_id = safe_str(row.get("target_global_id"))
        if source_id not in node_lookup or target_id not in node_lookup:
            continue
        source_idx = node_lookup[source_id]
        target_idx = node_lookup[target_id]
        source_type = safe_str(nodes.at[source_idx, "node_type"]) or "unknown"
        target_type = safe_str(nodes.at[target_idx, "node_type"]) or "unknown"
        family = safe_str(row.get("edge_family")) or "unknown"
        forward_relation = f"{source_type}__{family}__{target_type}__fwd"
        forward_relation_id = relation_map.setdefault(forward_relation, len(relation_map))

        src_vec = alignment_source[source_idx] if alignment_source.size else np.asarray([], dtype=np.float32)
        tgt_vec = alignment_source[target_idx] if alignment_source.size else np.asarray([], dtype=np.float32)
        omega, provenance, alignment, variance_score = confidence_components(src_vec, tgt_vec, row)

        add_edge_row(
            edge_rows,
            source_idx,
            target_idx,
            source_id,
            target_id,
            forward_relation,
            forward_relation_id,
            row,
            "forward",
            omega,
            provenance,
            alignment,
            variance_score,
            confidence_threshold,
        )
        edge_index_pairs.append((source_idx, target_idx))
        edge_types.append(forward_relation_id)
        edge_weights.append(omega)
        edge_attrs.append([omega, provenance, alignment, variance_score])

        if not bool(row.get("directed", False)):
            reverse_relation = f"{target_type}__{family}__{source_type}__rev"
            reverse_relation_id = relation_map.setdefault(reverse_relation, len(relation_map))
            add_edge_row(
                edge_rows,
                target_idx,
                source_idx,
                target_id,
                source_id,
                reverse_relation,
                reverse_relation_id,
                row,
                "reverse",
                omega,
                provenance,
                alignment,
                variance_score,
                confidence_threshold,
            )
            edge_index_pairs.append((target_idx, source_idx))
            edge_types.append(reverse_relation_id)
            edge_weights.append(omega)
            edge_attrs.append([omega, provenance, alignment, variance_score])

    edge_frame = pd.DataFrame(edge_rows)
    edge_index = np.asarray(edge_index_pairs, dtype=np.int64).T if edge_index_pairs else np.empty((2, 0), dtype=np.int64)
    edge_type = np.asarray(edge_types, dtype=np.int64) if edge_types else np.empty((0,), dtype=np.int64)
    edge_weight = np.asarray(edge_weights, dtype=np.float32) if edge_weights else np.empty((0,), dtype=np.float32)
    edge_attr = np.asarray(edge_attrs, dtype=np.float32) if edge_attrs else np.empty((0, 4), dtype=np.float32)

    feature_slices: dict[str, dict[str, int]] = {}
    start = 0
    for name, block in [
        ("numeric", numeric_scaled),
        ("categorical", categorical_matrix),
        ("text_hash", text_matrix),
        ("semantic", semantic_matrix),
    ]:
        if block.size == 0:
            continue
        feature_slices[name] = {"start": start, "end": start + block.shape[1]}
        start += block.shape[1]

    summary = {
        "node_count": int(len(nodes)),
        "edge_count": int(len(edge_frame)),
        "relation_count": int(len(relation_map)),
        "feature_dim": int(x.shape[1]),
        "text_hash_dim": int(text_hash_dim),
        "semantic_embedding_dim": int(semantic_embedding_dim),
        "semantic_embedding_coverage_rate": float(semantic_coverage.mean()) if len(semantic_coverage) else 0.0,
        "low_confidence_edge_count": int(edge_frame[edge_frame["low_confidence"]].shape[0]) if not edge_frame.empty else 0,
        "low_confidence_threshold": float(confidence_threshold),
        "node_types": sorted({safe_str(value) for value in nodes["node_type"].tolist()}),
        "anonymized_node_types": sorted(anonymize_types),
        "sensitive_columns_used": sensitive_columns,
        "feature_slices": feature_slices,
        "stage_notes": {
            "stage_5": "Semantic enrichment and vector alignment were applied through text hashing and optional external embeddings.",
            "stage_6": "Graph machine learning tensors were prepared for heterogenous message passing frameworks such as R-GCN and HAN.",
        },
        "uncertainty_formula": "omega = 0.45*provenance + 0.35*alignment + 0.20*variance_stability",
        "privacy_notes": [
            "Citizen-like node types can be anonymized through the anonymize-node-types flag.",
            "Sensitive columns are orthogonally projected out of narrative vectors when present.",
        ],
    }

    return node_features, edge_frame, x, edge_index, edge_type, edge_weight, summary, relation_map, feature_slices, edge_attr


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GNN-ready tensors, relation metadata, and uncertainty-aware edge confidence scores")
    parser.add_argument("--text-hash-dim", type=int, default=TEXT_HASH_DIM)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--anonymize-node-types", type=str, default=",".join(sorted(DEFAULT_ANON_TYPES)))
    parser.add_argument("--sensitive-columns", type=str, default="")
    args = parser.parse_args()

    ensure_output_dirs()
    GNN_DIR.mkdir(parents=True, exist_ok=True)

    nodes, edges = load_nodes_edges()
    if nodes.empty or edges.empty:
        raise FileNotFoundError("nodes.csv and edges.csv are required before GNN preparation can run.")

    embedding_candidates = [
        ANALYSIS_DIR / "node_semantic_embeddings.parquet",
        ANALYSIS_DIR / "document_embeddings.parquet",
        DATA_DIR / "document_embeddings.parquet",
        DATA_DIR / "node_semantic_embeddings.parquet",
    ]
    semantic_embedding_map, semantic_embedding_dim, embedding_source = load_embedding_map(embedding_candidates)

    sensitive_columns = [column.strip() for column in args.sensitive_columns.split(",") if column.strip()]
    if not sensitive_columns:
        sensitive_columns = auto_sensitive_columns(nodes)

    anonymize_types = {value.strip().lower() for value in args.anonymize_node_types.split(",") if value.strip()}
    node_features, edge_frame, x, edge_index, edge_type, edge_weight, summary, relation_map, feature_slices, edge_attr = build_gnn_outputs(
        nodes=nodes,
        edges=edges,
        semantic_embedding_map=semantic_embedding_map,
        semantic_embedding_dim=semantic_embedding_dim,
        anonymize_types=anonymize_types,
        sensitive_columns=sensitive_columns,
        text_hash_dim=args.text_hash_dim,
        confidence_threshold=float(args.confidence_threshold),
    )

    summary["semantic_embedding_source"] = embedding_source
    summary["confidence_threshold"] = float(args.confidence_threshold)
    summary["pipeline_stages"] = [
        "Stage 1: Source Recovery & JSON Validation",
        "Stage 2: Relational Tabular Extraction",
        "Stage 3: Multimodal Edge Reconstruction",
        "Stage 4: Graph Normalization",
        "Stage 5: Semantic Enrichment & Vector Alignment",
        "Stage 6: Graph Machine Learning Preparation",
    ]

    feature_frame = node_features.copy()
    feature_frame["node_index"] = np.arange(len(feature_frame), dtype=int)
    feature_frame["feature_dim"] = int(x.shape[1])
    feature_frame.to_csv(GNN_DIR / "node_features.csv", index=False)
    edge_frame.to_csv(GNN_DIR / "edge_features.csv", index=False)

    np.save(GNN_DIR / "node_features.npy", x)
    np.save(GNN_DIR / "edge_index.npy", edge_index)
    np.save(GNN_DIR / "edge_type.npy", edge_type)
    np.save(GNN_DIR / "edge_weight.npy", edge_weight)
    np.save(GNN_DIR / "edge_attr.npy", edge_attr)

    node_index_frame = pd.DataFrame(
        {
            "node_index": np.arange(len(node_features), dtype=int),
            "global_id": node_features["global_id"].astype(str),
            "public_label": node_features["public_label"].astype(str),
            "node_type": node_features["node_type"].astype(str),
            "methodological_phase": node_features["methodological_phase"].astype(str),
            "is_anonymized": node_features["is_anonymized"].astype(bool),
        }
    )
    node_index_frame.to_csv(GNN_DIR / "node_index.csv", index=False)

    (GNN_DIR / "relation_map.json").write_text(json.dumps(relation_map, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    (GNN_DIR / "feature_slices.json").write_text(json.dumps(feature_slices, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    (GNN_DIR / "gnn_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    np.savez_compressed(
        GNN_DIR / "gnn_dataset.npz",
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        edge_attr=edge_attr,
        node_ids=node_features["global_id"].astype(str).to_numpy(),
    )

    print(f"GNN-ready data written to {GNN_DIR}")
    print(f"Nodes: {len(node_features)} | Edges: {len(edge_frame)} | Relations: {len(relation_map)}")
    print(f"Feature dim: {x.shape[1]} | Low-confidence threshold: {args.confidence_threshold}")


if __name__ == "__main__":
    main()

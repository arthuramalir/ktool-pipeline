from __future__ import annotations

import os
import json
from pathlib import Path

import networkx as nx
import pandas as pd

from graph_utils import build_graph, load_nodes_edges, safe_centrality, simple_graph, write_frame, write_json, ANALYSIS_DIR

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173_synthetic")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")


def normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if maximum == minimum:
        return pd.Series([0.0] * len(numeric), index=series.index)
    return (numeric - minimum) / (maximum - minimum)


def node_label(row: pd.Series) -> str:
    for column in ["label", "title", "name", "public_label", "global_id"]:
        value = str(row.get(column, "")).strip()
        if value and value.lower() not in {"nan", "none", "unnamed"}:
            return value
    return str(row.get("global_id", "Unknown"))


def count_pipe_items(value) -> int:
    if pd.isna(value) or not str(value).strip():
        return 0
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return len(parsed)
        except Exception:
            return 1
    return len([part for part in text.replace("|", ",").split(",") if part.strip()])


def run_candidates() -> pd.DataFrame:
    nodes_df, edges_df = load_nodes_edges()
    if nodes_df.empty:
        print("[ERROR] Synthetic nodes table is empty.")
        return pd.DataFrame()

    graph = simple_graph(build_graph(directed=False))
    metrics = safe_centrality(graph)
    if metrics.empty:
        metrics = pd.DataFrame(columns=["global_id", "degree", "degree_centrality", "betweenness_centrality", "closeness_centrality"])

    candidate_nodes = nodes_df[nodes_df["node_type"].astype(str).str.lower().eq("project")].copy()
    if candidate_nodes.empty:
        print("[ERROR] No project nodes found for prototype suggestion scoring.")
        return pd.DataFrame()

    candidate_nodes = candidate_nodes.merge(metrics, on="global_id", how="left")
    candidate_nodes["associated_budget"] = pd.to_numeric(candidate_nodes.get("associated_budget", 0), errors="coerce").fillna(0.0)
    candidate_nodes["degree"] = pd.to_numeric(candidate_nodes.get("degree", 0), errors="coerce").fillna(0.0)
    candidate_nodes["betweenness_centrality"] = pd.to_numeric(candidate_nodes.get("betweenness_centrality", 0), errors="coerce").fillna(0.0)
    candidate_nodes["perception_count"] = candidate_nodes.get("perception_ids", pd.Series(index=candidate_nodes.index, dtype=str)).apply(count_pipe_items)
    candidate_nodes["agent_count"] = candidate_nodes.get("agent_ids", pd.Series(index=candidate_nodes.index, dtype=str)).apply(count_pipe_items)
    candidate_nodes["topic_count"] = candidate_nodes.get("topic_ids", pd.Series(index=candidate_nodes.index, dtype=str)).apply(count_pipe_items)

    budget_score = normalize(candidate_nodes["associated_budget"])
    bridge_score = normalize(candidate_nodes["betweenness_centrality"])
    perception_score = normalize(candidate_nodes["perception_count"])
    agent_score = normalize(candidate_nodes["agent_count"])
    topic_score = normalize(candidate_nodes["topic_count"])

    candidate_nodes["prototype_candidate_score"] = (
        0.32 * budget_score
        + 0.28 * bridge_score
        + 0.16 * perception_score
        + 0.12 * agent_score
        + 0.12 * topic_score
    )
    candidate_nodes = candidate_nodes.sort_values("prototype_candidate_score", ascending=False)

    rows = []
    for _, row in candidate_nodes.iterrows():
        rows.append(
            {
                "global_id": row["global_id"],
                "label": node_label(row),
                "node_type": row.get("node_type", "project"),
                "associated_budget": int(row.get("associated_budget", 0) or 0),
                "impact_level": str(row.get("impact_level", "")),
                "degree": int(row.get("degree", 0) or 0),
                "betweenness_centrality": round(float(row.get("betweenness_centrality", 0) or 0), 6),
                "perception_count": int(row.get("perception_count", 0) or 0),
                "agent_count": int(row.get("agent_count", 0) or 0),
                "topic_count": int(row.get("topic_count", 0) or 0),
                "prototype_candidate_score": round(float(row.get("prototype_candidate_score", 0) or 0), 6),
                "suggestion_reason": "High structural reach with visible perception and agent coverage." if row.get("prototype_candidate_score", 0) >= 0.6 else "Potentially useful but needs more manual review.",
            }
        )

    df = pd.DataFrame(rows)
    write_frame(df, "prototype_project_candidates.csv")
    write_json(ANALYSIS_DIR / "prototype_project_candidates_summary.json", {
        "platform_id": PLATFORM_ID,
        "output_subdir": OUTPUT_SUBDIR,
        "candidate_count": int(len(df)),
        "top_candidate": df.head(1).to_dict("records") if not df.empty else [],
    })
    print(f"Saved prototype_project_candidates.csv with {len(df)} candidates")
    return df


if __name__ == "__main__":
    run_candidates()

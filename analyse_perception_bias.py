from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from strapi_utils import data_dir


def parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            return [value]
    return [str(value)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse bias and representation across perception clusters")
    parser.add_argument("--documents", type=Path, default=data_dir() / "documents.parquet")
    parser.add_argument("--consensus", type=Path, default=data_dir() / "consensus_perception_space.parquet")
    parser.add_argument("--tfidf", type=Path, default=data_dir() / "tfidf_matrix.npz")
    parser.add_argument("--tfidf-vocabulary", type=Path, default=data_dir() / "tfidf_vocabulary.json")
    parser.add_argument("--graph-analysis", type=Path, default=data_dir() / "actor_perception_analysis.parquet")
    parser.add_argument("--output", type=Path, default=data_dir() / "bias_analysis.csv")
    args = parser.parse_args()

    documents = pd.read_parquet(args.documents)
    consensus = pd.read_parquet(args.consensus)
    documents["entity_type"] = documents["entity_type"].astype(str)
    cluster_map = consensus.set_index("document_id")["cluster_consensus_id"].to_dict()
    documents["cluster_consensus_id"] = documents["id"].astype(str).map(cluster_map)

    rows: list[dict[str, Any]] = []

    # Actor representation bias: which actors dominate clusters?
    for cluster_id, subset in documents.groupby("cluster_consensus_id"):
        agent_counts = Counter()
        for _, row in subset.iterrows():
            for agent in parse_json_list(row.get("related_agents")):
                agent_counts[agent] += 1
        total = sum(agent_counts.values()) or 1
        if agent_counts:
            actor_id, count = agent_counts.most_common(1)[0]
            rows.append(
                {
                    "analysis_type": "actor_representation_bias",
                    "subject": f"cluster_{int(cluster_id)}",
                    "metric": "top_actor_share",
                    "value": float(count / total),
                    "details_json": json.dumps({"top_actor": actor_id, "count": count, "total": total}, ensure_ascii=True),
                }
            )

    # Semantic isolation: actors with narrow, highly self-contained narratives.
    actor_docs = documents[documents["entity_type"].isin(["agent", "organization", "organisation"])]
    for actor_id, subset in actor_docs.groupby("entity_id"):
        cluster_diversity = subset["cluster_consensus_id"].nunique(dropna=True)
        isolated_score = 1.0 / max(1, cluster_diversity)
        rows.append(
            {
                "analysis_type": "semantic_isolation",
                "subject": f"actor_{actor_id}",
                "metric": "cluster_diversity_inverse",
                "value": float(isolated_score),
                "details_json": json.dumps({"cluster_diversity": int(cluster_diversity), "document_count": int(len(subset))}, ensure_ascii=True),
            }
        )

    # Missing bridges: graph communities with little semantic spread.
    if args.graph_analysis.exists():
        graph_analysis = pd.read_parquet(args.graph_analysis)
        if "node_id" in graph_analysis.columns:
            merged = graph_analysis.merge(consensus, left_on="node_id", right_on="document_id", how="left")
            for community_id, subset in merged.groupby("graph_community"):
                unique_clusters = subset["cluster_consensus_id"].nunique(dropna=True)
                bridge_score = float(subset["betweenness_centrality"].fillna(0).mean())
                rows.append(
                    {
                        "analysis_type": "missing_bridges",
                        "subject": f"graph_community_{int(community_id)}",
                        "metric": "semantic_cluster_count",
                        "value": float(unique_clusters),
                        "details_json": json.dumps({"mean_betweenness": bridge_score}, ensure_ascii=True),
                    }
                )

    # Dominant narratives: largest semantic footprints based on TF-IDF terms.
    tfidf = load_npz(args.tfidf)
    vocabulary = json.loads(args.tfidf_vocabulary.read_text(encoding="utf-8"))
    terms = [term for term, idx in sorted(vocabulary.items(), key=lambda item: item[1])]
    weights = np.asarray(tfidf.mean(axis=0)).ravel()
    top_idx = np.argsort(weights)[::-1][:50]
    for rank, idx in enumerate(top_idx[:20], start=1):
        if weights[idx] <= 0:
            continue
        rows.append(
            {
                "analysis_type": "dominant_narrative",
                "subject": terms[idx],
                "metric": "mean_tfidf_weight",
                "value": float(weights[idx]),
                "details_json": json.dumps({"rank": rank}, ensure_ascii=True),
            }
        )

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote perception bias analysis to {args.output}")


if __name__ == "__main__":
    main()

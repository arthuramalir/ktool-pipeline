"""Link story clusters to claims via information_id to source_node_id.

Creates a cluster_claim_matrix.csv that lets human profile builders
browse clusters filtered by value dimension, see claim density, and
spot internal contradictions all before writing a single profile.

Usage:
    set KTOOL_PLATFORM_ID=173_synthetic & set KTOOL_OUTPUT_SUBDIR=test
    python src/analysis/23_link_clusters_to_claims.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173_synthetic")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
ANALYSIS_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR / "analysis"
NARRATIVE_DIR = ANALYSIS_DIR / "narrative_layers"


def load_csv_or_empty(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def main() -> None:
    print(f"Linking clusters to claims - {PLATFORM_ID}/{OUTPUT_SUBDIR}")

    clusters = load_csv_or_empty(ANALYSIS_DIR / "quote_clusters.csv")
    if clusters.empty:
        print("ERROR: quote_clusters.csv not found. Run 17_cluster_quotes_into_profiles.py first.")
        return
    print(f"  Clusters: {len(clusters)} rows across {clusters['cluster_id'].nunique()} clusters")

    claims = load_csv_or_empty(NARRATIVE_DIR / "claim_nodes.csv")
    if claims.empty:
        print("ERROR: claim_nodes.csv not found. Run 21_extract_narrative_layers.py first.")
        return
    print(f"  Claims: {len(claims)} claim nodes")

    clus = clusters[["information_id", "cluster_id", "quote", "channel_code", "topics_thematic_areas", "values"]].copy()
    clus["information_id"] = clus["information_id"].astype(str).str.strip()

    clm = claims[["global_id", "source_node_id", "source_node_type", "narrative_level", "value_dimension", "belief_level", "negated", "conditional", "emergency_frame"]].copy()
    clm["source_node_id"] = clm["source_node_id"].astype(str).str.strip()

    matrix = clus.merge(clm, left_on="information_id", right_on="source_node_id", how="left")
    matrix.rename(columns={"global_id": "claim_id"}, inplace=True)

    has_val = matrix["value_dimension"].notna() & (matrix["value_dimension"] != "")
    cluster_dims = matrix[has_val].groupby("cluster_id")["value_dimension"].apply(set)
    contradiction_clusters = cluster_dims[cluster_dims.apply(len) > 1]
    matrix["contradiction_flag"] = matrix["cluster_id"].isin(contradiction_clusters.index)

    summary_rows = []
    for cid, group in matrix.groupby("cluster_id"):
        n_quotes = group["quote"].dropna().nunique()
        n_claims = group["claim_id"].dropna().nunique()
        group_has_val = group["value_dimension"].notna() & (group["value_dimension"] != "")
        dims = group[group_has_val].groupby("value_dimension")["claim_id"].nunique()
        dim_str = " | ".join(f"{d} ({c})" for d, c in dims.items()) if not dims.empty else ""
        summary_rows.append({
            "cluster_id": cid,
            "quote_count": n_quotes,
            "claim_count": n_claims,
            "value_dimensions": dim_str,
            "dominant_dimension": dims.idxmax() if not dims.empty else "",
            "has_contradiction": cid in contradiction_clusters.index,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("cluster_id")

    matrix.to_csv(ANALYSIS_DIR / "cluster_claim_matrix.csv", index=False)
    summary_df.to_csv(ANALYSIS_DIR / "cluster_claim_summary.csv", index=False)
    print(f"  Wrote cluster_claim_matrix.csv ({len(matrix)} rows)")
    print(f"  Wrote cluster_claim_summary.csv ({len(summary_df)} rows)")
    print(f"  Contradictory clusters: {contradiction_clusters.shape[0]}")
    n_linked = matrix["claim_id"].notna().sum()
    print(f"  Coverage: {n_linked}/{len(claims)} claims linked to a cluster")


if __name__ == "__main__":
    main()

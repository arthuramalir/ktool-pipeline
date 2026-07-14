"""Cluster-level narrative simulation using FJ opinion dynamics.

Treats each story cluster as an opinion-holding node. Cluster opinions are
aggregated from claims' value dimensions. Cross-cluster influence flows
through semantic edges between quotes in different clusters.

Usage:
    set KTOOL_PLATFORM_ID=173_synthetic & set KTOOL_OUTPUT_SUBDIR=test
    python src/analysis/27_cluster_narrative_simulation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = "173_synthetic"
OUTPUT_SUBDIR = "test"
ANALYSIS_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR / "analysis"
NARRATIVE_DIR = ANALYSIS_DIR / "narrative_layers"

VD_ORDER = [
    "cultural_identity", "social_justice", "collaboration",
    "innovation_drive", "evidence_based", "community_autonomy",
    "austerity_scarcity",
]
M = len(VD_ORDER)

MAX_ITER = 300
TOL = 1e-5


def load_csv_or_empty(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def main() -> None:
    print(f"Cluster narrative simulation — {PLATFORM_ID}/{OUTPUT_SUBDIR}")

    # 1. Load cluster-claim matrix
    matrix = load_csv_or_empty(ANALYSIS_DIR / "cluster_claim_matrix.csv")
    if matrix.empty:
        print("ERROR: cluster_claim_matrix.csv not found. Run 23_link_clusters_to_claims.py first.")
        return

    # 2. Build per-cluster opinion vectors from claim value dimensions
    has_val = matrix["value_dimension"].notna() & (matrix["value_dimension"] != "")
    valid = matrix[has_val].copy()
    # Count claims per value dimension per cluster
    cluster_dim = valid.groupby(["cluster_id", "value_dimension"]).size().reset_index(name="claim_count")
    clusters = sorted(matrix["cluster_id"].unique())

    opinion = np.zeros((len(clusters), M), dtype=np.float64)
    claim_counts = np.zeros(len(clusters), dtype=int)
    for i, cid in enumerate(clusters):
        sub = cluster_dim[cluster_dim["cluster_id"] == cid]
        for _, r in sub.iterrows():
            dim = r["value_dimension"]
            if dim in VD_ORDER:
                idx = VD_ORDER.index(dim)
                opinion[i, idx] = float(r["claim_count"])
                claim_counts[i] += int(r["claim_count"])
        # Normalize to unit vector per cluster
        norm = np.linalg.norm(opinion[i])
        if norm > 0:
            opinion[i] /= norm

    print(f"\n  Cluster opinions ({len(clusters)} clusters, {M} dimensions):")
    for i, cid in enumerate(clusters):
        top = sorted(zip(VD_ORDER, opinion[i]), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{d}={v:.3f}" for d, v in top if v > 0)
        print(f"    {cid}: [{top_str}] ({claim_counts[i]} claims)")

    # 3. Build cluster adjacency from cross-cluster semantic edges
    quote_clusters = load_csv_or_empty(ANALYSIS_DIR / "quote_clusters.csv")
    semantic_edges = load_csv_or_empty(ANALYSIS_DIR / "quote_semantic_edges.csv")

    if quote_clusters.empty or semantic_edges.empty:
        print("ERROR: clusters or semantic edges missing.")
        return

    q_map = quote_clusters[["information_id", "cluster_id"]].drop_duplicates()
    info_col = "source_global_id"
    sem = semantic_edges.merge(q_map, left_on=info_col, right_on="information_id", how="inner")
    sem = sem.rename(columns={"cluster_id": "src_cluster"})
    sem = sem.merge(q_map, left_on="target_global_id", right_on="information_id", how="inner", suffixes=("", "_tgt"))
    sem = sem.rename(columns={"cluster_id": "tgt_cluster"})
    cross = sem[sem["src_cluster"].notna() & sem["tgt_cluster"].notna() & (sem["src_cluster"] != sem["tgt_cluster"])]

    adj = np.zeros((len(clusters), len(clusters)), dtype=np.float64)
    cid_to_idx = {cid: i for i, cid in enumerate(clusters)}
    for _, r in cross.iterrows():
        i = cid_to_idx.get(r["src_cluster"])
        j = cid_to_idx.get(r["tgt_cluster"])
        if i is not None and j is not None:
            w = float(r.get("weight", 1.0) or 1.0)
            adj[i, j] += w
            adj[j, i] += w

    # Row-normalize adjacency
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    adj_norm = adj / row_sums
    print(f"\n  Cross-cluster edges: {len(cross)}")
    print(f"  Adjacency density: {np.mean(adj > 0):.1%}")

    # 4. FJ dynamics: u(t+1) = diag(Lambda) * u0 + (I - diag(Lambda)) * A * u(t)
    # Lambda = stubbornness; more claims = more stubborn
    max_claims = max(claim_counts.max(), 1)
    lam_diag = np.array([min(0.9, c / max_claims * 0.8 + 0.1) for c in claim_counts], dtype=np.float64)
    Lambda = np.diag(lam_diag)

    u = opinion.copy()
    for iteration in range(MAX_ITER):
        u_prev = u.copy()
        u = Lambda @ opinion + (np.eye(len(clusters)) - Lambda) @ adj_norm @ u
        if np.max(np.abs(u - u_prev)) < TOL:
            break
    baseline = u.copy()
    print(f"\n  Converged in {iteration + 1} iterations")

    # Save baseline
    baseline_rows = []
    for i, cid in enumerate(clusters):
        baseline_rows.append({
            "cluster_id": cid,
            **{dim: round(float(baseline[i, j]), 6) for j, dim in enumerate(VD_ORDER)},
            "claim_count": int(claim_counts[i]),
            "stubbornness": round(float(lam_diag[i]), 4),
        })
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(ANALYSIS_DIR / "cluster_narrative_baseline.csv", index=False)
    print(f"  Saved: cluster_narrative_baseline.csv")

    # Save adjacency
    adj_rows = []
    for i, cid in enumerate(clusters):
        for j, cid2 in enumerate(clusters):
            if adj[i, j] > 0 and i < j:
                adj_rows.append({
                    "source_cluster": cid,
                    "target_cluster": cid2,
                    "weight": round(float(adj[i, j]), 4),
                })
    adj_df = pd.DataFrame(adj_rows).sort_values("weight", ascending=False)
    adj_df.to_csv(ANALYSIS_DIR / "cluster_narrative_adjacency.csv", index=False)
    print(f"  Saved: cluster_narrative_adjacency.csv ({len(adj_df)} edges)")

    # 5. Interventions: connect an external agent to a target cluster
    # Agent opinion = uniform across all dimensions (neutral)
    # Simulate for each cluster
    intervention_rows = []
    agent_opinion = np.ones(M, dtype=np.float64) / np.sqrt(M)

    for target_idx, target_cid in enumerate(clusters):
        # Add agent as new node with low stubbornness
        N_aug = len(clusters) + 1
        u0_aug = np.vstack([opinion, agent_opinion.reshape(1, -1)])
        lam_aug = np.concatenate([lam_diag, [0.2]])  # agent is easy to influence
        Lambda_aug = np.diag(lam_aug)

        A_aug = np.zeros((N_aug, N_aug), dtype=np.float64)
        A_aug[:len(clusters), :len(clusters)] = adj_norm
        # Agent connects to target cluster (bidirectional)
        A_aug[target_idx, len(clusters)] = 1.0
        A_aug[len(clusters), target_idx] = 1.0
        row_sums_aug = A_aug.sum(axis=1, keepdims=True)
        row_sums_aug = np.where(row_sums_aug == 0, 1.0, row_sums_aug)
        A_aug_norm = A_aug / row_sums_aug

        u_a = u0_aug.copy()
        for _ in range(MAX_ITER):
            u_prev = u_a.copy()
            u_a = Lambda_aug @ u0_aug + (np.eye(N_aug) - Lambda_aug) @ A_aug_norm @ u_a
            if np.max(np.abs(u_a - u_prev)) < TOL:
                break

        # Compute deltas for all clusters
        for i, cid in enumerate(clusters):
            delta = u_a[i] - baseline[i]
            row = {
                "target_cluster": target_cid,
                "affected_cluster": cid,
                **{f"delta_{dim}": round(float(delta[j]), 6) for j, dim in enumerate(VD_ORDER)},
                "max_abs_delta": round(float(np.max(np.abs(delta))), 6),
            }
            intervention_rows.append(row)

    intervention_df = pd.DataFrame(intervention_rows)
    intervention_df.to_csv(ANALYSIS_DIR / "cluster_narrative_interventions.csv", index=False)
    print(f"  Saved: cluster_narrative_interventions.csv ({len(intervention_df)} rows)")

    summary = {
        "n_clusters": len(clusters),
        "n_dimensions": M,
        "n_cross_cluster_edges": len(cross),
        "adjacency_density": round(float(np.mean(adj > 0)), 4),
        "dimensions": VD_ORDER,
    }
    with open(ANALYSIS_DIR / "cluster_narrative_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: cluster_narrative_summary.json")

    print(f"\n  Top intervention deltas:")
    top_deltas = intervention_df.loc[intervention_df.groupby("target_cluster")["max_abs_delta"].idxmax()]
    for _, r in top_deltas.iterrows():
        print(f"    Agent -> {r['target_cluster']}: {r['affected_cluster']} delta={r['max_abs_delta']:.6f}")


if __name__ == "__main__":
    main()

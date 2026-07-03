"""Link Intervention Simulation — maps structural impact on perception space.

For each GNN-proposed link, simulates adding it to the graph and measures:
  - PageRank shift of perception nodes (narrative influence change)
  - Whether the link merges disconnected components
  - How many information nodes become reachable from each perception
  - Composite sensitivity score

Usage:
  set KTOOL_PLATFORM_ID=173 & set KTOOL_OUTPUT_SUBDIR=test
  python src/analysis/19_link_intervention_simulation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from analysis.graph_utils import ANALYSIS_DIR, ANALYTICS_DIR, DATA_DIR, write_frame

PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
DATA_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR

NODES_PATH = DATA_DIR / "analytics" / "nodes.csv"
EDGES_PATH = DATA_DIR / "analytics" / "edges.csv"
GNN_RECS_PATH = ANALYSIS_DIR / "gnn" / "gnn_link_recommendations.csv"
PERCEPTION_DIAG_PATH = ANALYSIS_DIR / "perception_diagnostics.csv"

TOP_K = 20
PR_ALPHA = 0.85
MAX_HOPS = 3


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 2:
        return pd.read_csv(path)
    return pd.DataFrame()


def main():
    nodes = load_csv(NODES_PATH)
    edges = load_csv(EDGES_PATH)
    recs = load_csv(GNN_RECS_PATH)

    if nodes.empty or edges.empty:
        print("ERROR: graph tables not found.")
        sys.exit(1)
    if recs.empty:
        print("ERROR: no GNN recommendations found.")
        sys.exit(1)

    print(f"Nodes: {len(nodes)}, Edges: {len(edges)}, Recommendations: {len(recs)}")

    G = nx.Graph()
    for _, row in nodes.iterrows():
        gid = str(row["global_id"])
        G.add_node(gid, type=str(row.get("node_type", "")), label=str(row.get("label", gid)))
    for _, row in edges.iterrows():
        G.add_edge(str(row["source_global_id"]), str(row["target_global_id"]))

    perception_ids = [n for n, d in G.nodes(data=True) if d.get("type") == "perception"]
    information_ids = {n for n, d in G.nodes(data=True) if d.get("type") == "information"}
    agent_ids = {n for n, d in G.nodes(data=True) if d.get("type") == "agent"}
    project_ids = {n for n, d in G.nodes(data=True) if d.get("type") == "project"}

    print(f"Perception nodes: {len(perception_ids)}, Information nodes: {len(information_ids)}")
    print(f"Graph components: {nx.number_connected_components(G)}")

    baseline_pr = nx.pagerank(G, alpha=PR_ALPHA)
    baseline_components = {i: set(c) for i, c in enumerate(nx.connected_components(G))}
    node_to_component = {}
    for cid, comp in baseline_components.items():
        for n in comp:
            node_to_component[n] = cid

    baseline_info_reach = {}
    for p in perception_ids:
        if p not in G:
            baseline_info_reach[p] = set()
            continue
        reached = set()
        seen = {p}
        frontier = {p}
        for _ in range(MAX_HOPS):
            next_frontier = set()
            for f in frontier:
                for nb in G.neighbors(f):
                    if nb not in seen:
                        seen.add(nb)
                        next_frontier.add(nb)
                        if nb in information_ids:
                            reached.add(nb)
            frontier = next_frontier
            if not frontier:
                break
        baseline_info_reach[p] = reached

    recs = recs.head(TOP_K).copy()
    results = []
    perception_impact_rows = []

    for idx, row in recs.iterrows():
        src = str(row.get("source_global_id", "")).strip()
        tgt = str(row.get("target_global_id", "")).strip()
        if not src or not tgt or src not in G or tgt not in G:
            continue

        G_sim = G.copy()
        G_sim.add_edge(src, tgt)

        sim_pr = nx.pagerank(G_sim, alpha=PR_ALPHA)

        perception_pr_deltas = {}
        for p in perception_ids:
            delta = sim_pr.get(p, 0.0) - baseline_pr.get(p, 0.0)
            perception_pr_deltas[p] = round(delta, 6)

        avg_perception_delta = float(np.mean(list(perception_pr_deltas.values()))) if perception_pr_deltas else 0.0
        n_gaining = sum(1 for v in perception_pr_deltas.values() if v > 0)
        n_losing = sum(1 for v in perception_pr_deltas.values() if v < 0)

        comp_src = node_to_component.get(src)
        comp_tgt = node_to_component.get(tgt)
        merges_components = (comp_src is not None and comp_tgt is not None and comp_src != comp_tgt)

        sim_components = {i: set(c) for i, c in enumerate(nx.connected_components(G_sim))}
        n_components_before = len(baseline_components)
        n_components_after = len(sim_components)
        components_merged = n_components_before - n_components_after

        info_reach_expansion = 0
        gainers = []
        for p in perception_ids:
            if p not in G_sim:
                continue
            if p not in G:
                baseline_set = set()
            else:
                baseline_set = baseline_info_reach.get(p, set())

            reached = set()
            seen = {p}
            frontier = {p}
            for _ in range(MAX_HOPS):
                next_frontier = set()
                for f in frontier:
                    for nb in G_sim.neighbors(f):
                        if nb not in seen:
                            seen.add(nb)
                            next_frontier.add(nb)
                            if nb in information_ids:
                                reached.add(nb)
                frontier = next_frontier
                if not frontier:
                    break

            new_reachable = reached - baseline_set
            if new_reachable:
                gainers.append((p, len(new_reachable)))
                info_reach_expansion += len(new_reachable)

        top_gainer = max(gainers, key=lambda x: x[1]) if gainers else ("", 0)
        max_delta_p = max(perception_pr_deltas, key=lambda k: abs(perception_pr_deltas[k])) if perception_pr_deltas else ""
        max_delta_val = perception_pr_deltas.get(max_delta_p, 0.0)

        sensitivity_score = round(
            abs(avg_perception_delta) * 10
            + (0.3 if merges_components else 0.0)
            + min(info_reach_expansion / max(1, len(information_ids)), 0.5)
            + min(abs(max_delta_val) * 50, 0.2),
            4,
        )

        label_src = row.get("source_label", src)
        label_tgt = row.get("target_label", tgt)
        link_type = row.get("link_type", f"{row.get('source_node_type','?')} -> {row.get('target_node_type','?')}")

        result = {
            "source_global_id": src,
            "target_global_id": tgt,
            "source_label": label_src,
            "target_label": label_tgt,
            "link_type": link_type,
            "link_probability": row.get("link_probability", ""),
            "merges_components": merges_components,
            "components_merged_count": components_merged,
            "avg_perception_PR_delta": avg_perception_delta,
            "n_perceptions_gaining": n_gaining,
            "n_perceptions_losing": n_losing,
            "max_perception_PR_delta": round(max_delta_val, 6),
            "max_delta_perception_id": max_delta_p,
            "info_reach_expansion": info_reach_expansion,
            "top_gainer_perception": top_gainer[0],
            "top_gainer_new_infos": top_gainer[1],
            "sensitivity_score": sensitivity_score,
        }
        results.append(result)

        for p in perception_ids:
            perception_impact_rows.append({
                "source_global_id": src,
                "target_global_id": tgt,
                "source_label": label_src,
                "target_label": label_tgt,
                "perception_id": p,
                "PR_delta": perception_pr_deltas.get(p, 0.0),
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("sensitivity_score", ascending=False)
        write_frame(results_df, "link_intervention_scores.csv")
        print(f"Written {len(results_df)} intervention scores.")

    impact_df = pd.DataFrame(perception_impact_rows)
    if not impact_df.empty:
        write_frame(impact_df, "link_intervention_perception_impact.csv")
        print(f"Written {len(impact_df)} perception impact rows.")

    summary = {
        "n_recommendations_simulated": len(results_df),
        "n_merging_components": int(results_df["merges_components"].sum()) if not results_df.empty else 0,
        "avg_sensitivity": float(results_df["sensitivity_score"].mean()) if not results_df.empty else 0.0,
        "max_sensitivity": float(results_df["sensitivity_score"].max()) if not results_df.empty else 0.0,
        "total_info_reach_expansion": int(results_df["info_reach_expansion"].sum()) if not results_df.empty else 0,
    }
    (ANALYSIS_DIR / "link_intervention_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Summary: {json.dumps(summary, indent=2)}")
    print("Done.")


if __name__ == "__main__":
    import json
    main()

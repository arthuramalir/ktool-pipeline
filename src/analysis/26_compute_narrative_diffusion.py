"""Compute PageRank and narrative diffusion bias for all nodes.

Output: {platform_id}_narrative_diffusion.csv used by the Listening tab
to show "Most influential voices" (top PageRank) and diffusion bias
(which voices get more/less attention than expected by degree).

Usage:
    set KTOOL_PLATFORM_ID=173_synthetic & set KTOOL_OUTPUT_SUBDIR=test
    python src/analysis/26_compute_narrative_diffusion.py
"""

from __future__ import annotations

import os
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173_synthetic")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
ANALYTICS_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR / "analytics"
ANALYSIS_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR / "analysis"


def safe_str(v: object) -> str:
    if pd.isna(v) or v is None:
        return ""
    s = str(v).strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    return s


def fallback_label(label: object, gid: object) -> str:
    lbl = safe_str(label)
    if lbl:
        return lbl
    g = safe_str(gid)
    return g[-20:] if g else "unknown"


def main() -> None:
    print(f"PageRank + Diffusion Bias - {PLATFORM_ID}/{OUTPUT_SUBDIR}")

    nodes = pd.read_csv(ANALYTICS_DIR / "nodes.csv")
    edges = pd.read_csv(ANALYTICS_DIR / "edges.csv")
    print(f"  Nodes: {len(nodes)}, Edges: {len(edges)}")

    G = nx.Graph()
    for _, row in nodes.iterrows():
        gid = safe_str(row.get("global_id"))
        if gid:
            G.add_node(gid, node_type=safe_str(row.get("node_type")), label=fallback_label(row.get("label"), gid), type=safe_str(row.get("type")))

    for _, row in edges.iterrows():
        s = safe_str(row.get("source_global_id"))
        t = safe_str(row.get("target_global_id"))
        if s and t and s != t:
            G.add_edge(s, t)

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if G.number_of_edges() == 0:
        print("  SKIP: no edges")
        return

    # 1. Standard PageRank for influence
    pr = nx.pagerank(G, alpha=0.85, max_iter=200)

    # 2. Diffusion bias: personalized PageRank from two largest agent types
    type_groups: dict[str, list[str]] = {}
    for n, d in G.nodes(data=True):
        if d.get("node_type") != "agent":
            continue
        atype = safe_str(d.get("type"))
        if not atype:
            atype = "unknown"
        type_groups.setdefault(atype, []).append(n)

    sorted_types = sorted(type_groups.items(), key=lambda x: -len(x[1]))
    if len(sorted_types) >= 2:
        type_a_name, type_a_nodes = sorted_types[0]
        type_b_name, type_b_nodes = sorted_types[1]
        print(f"  Contrasting: {type_a_name} ({len(type_a_nodes)}) vs {type_b_name} ({len(type_b_nodes)})")

        personal_a = {n: 1.0 / len(type_a_nodes) for n in type_a_nodes}
        personal_b = {n: 1.0 / len(type_b_nodes) for n in type_b_nodes}

        valid_a = [n for n in personal_a if n in G]
        valid_b = [n for n in personal_b if n in G]

        pr_a = nx.pagerank(G, personalization={n: personal_a[n] for n in valid_a}, alpha=0.85, max_iter=200) if valid_a else {}
        pr_b = nx.pagerank(G, personalization={n: personal_b[n] for n in valid_b}, alpha=0.85, max_iter=200) if valid_b else {}
    else:
        type_a_name, type_b_name = "", ""
        pr_a, pr_b = {}, {}
        print("  SKIP: personalized PageRank (need 2+ agent types)")

    records = []
    all_nodes = set(pr.keys()) | set(pr_a.keys()) | set(pr_b.keys())
    for n in all_nodes:
        base_pr = pr.get(n, 0.0)
        a = pr_a.get(n, 0.0)
        b = pr_b.get(n, 0.0)
        diff = (a - b) / max(a + b, 1e-10) if pr_a or pr_b else 0.0
        nd = G.nodes[n]
        label_val = nd.get("label", n)
        records.append({
            "global_id": n,
            "label": str(label_val)[:60],
            "node_type": nd.get("node_type", ""),
            "pagerank": round(base_pr, 8),
            "agent_type": safe_str(nd.get("type")) if nd.get("node_type") == "agent" else "",
            "diffusion_bias": round(diff, 4),
        })

    df = pd.DataFrame(records).sort_values("pagerank", ascending=False)
    out_path = ANALYSIS_DIR / f"{PLATFORM_ID}_narrative_diffusion.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name} ({len(df)} rows)")

    top_pr = df.head(10)
    print(f"\n  Top 10 by PageRank:")
    for _, r in top_pr.iterrows():
        print(f"    {r['label'][:50]:50s} PR={r['pagerank']:.6f}  type={r['node_type']}")

    if pr_a or pr_b:
        top_bias = df.nlargest(5, "diffusion_bias")
        bottom_bias = df.nsmallest(5, "diffusion_bias")
        print(f"\n  Most biased toward {type_a_name} (positive):")
        for _, r in top_bias.iterrows():
            print(f"    {r['label'][:50]:50s} bias={r['diffusion_bias']:.3f}  type={r['node_type']}")
        print(f"\n  Most biased toward {type_b_name} (negative):")
        for _, r in bottom_bias.iterrows():
            print(f"    {r['label'][:50]:50s} bias={r['diffusion_bias']:.3f}  type={r['node_type']}")


if __name__ == "__main__":
    main()

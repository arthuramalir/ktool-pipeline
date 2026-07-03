"""Graph-only analyses for Platform 173 (Ireland).
Three analyses that REQUIRE graph structure — impossible on flat records.

Uses agent TYPE (NGO, public_administration, civil_society) instead of
investment labels used in the Platform 10 version.

Usage:
  set KTOOL_PLATFORM_ID=173 & set KTOOL_OUTPUT_SUBDIR=test
  python src/analysis/graph_only_analyses_173.py
"""

import os, sys, json
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_utils import build_graph, simple_graph, ANALYSIS_DIR, write_frame, write_json

PLOTS_DIR = ANALYSIS_DIR / "graph_only_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    pass

COLORS = ["#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261","#6d597a"]


def node_label(G, n, maxlen=60):
    d = G.nodes[n]
    ntype = d.get("node_type", "")
    if ntype == "information":
        txt = d.get("quote", d.get("description", ""))
        return str(txt)[:maxlen]
    return str(d.get("label", d.get("name", d.get("title", n))))[:maxlen]


# ===================================================================
# ANALYSIS 1: Agent Neighbourhood by TYPE (Graph Multi-hop)
# ===================================================================
#   WHY GRAPH? Measures which perception neighbourhoods different
#   agent types (NGO, public_administration, etc.) reach via
#   multi-hop graph traversal. Impossible with flat tables.
# ===================================================================

HOP_CAP = 3

def run_agent_neighbourhood_profiling(G_multi):
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Agent Neighbourhood by Type")
    print("Graph-only: multi-hop traversal through heterogeneous edges")
    print("=" * 60)

    G = simple_graph(G_multi)

    # Group agents by type
    agents_by_type = {}
    for n, d in G.nodes(data=True):
        if d.get("node_type") != "agent":
            continue
        atype = d.get("type", "unknown")
        if pd.isna(atype) or str(atype).strip() == "":
            atype = "unknown"
        agents_by_type.setdefault(str(atype), []).append(n)

    print(f"  Agent types with members:")
    for atype, nodes in sorted(agents_by_type.items(), key=lambda x: -len(x[1])):
        print(f"    {atype}: {len(nodes)} agents")

    # For each agent, find perceptions within HOP_CAP hops
    agent_perceptions = {}
    for n, d in G.nodes(data=True):
        if d.get("node_type") != "agent":
            continue
        visited = {n}
        frontier = {n}
        perceptions = set()
        for _ in range(HOP_CAP):
            next_f = set()
            for f in frontier:
                for nb in G.neighbors(f):
                    if nb in visited:
                        continue
                    ntype = G.nodes[nb].get("node_type", "")
                    if ntype == "agent":
                        continue
                    if ntype == "perception":
                        perceptions.add(nb)
                    visited.add(nb)
                    next_f.add(nb)
            frontier = next_f
        if perceptions:
            agent_perceptions[n] = perceptions

    print(f"  Agents with reachable perceptions: {len(agent_perceptions)}")
    all_percs = set()
    for v in agent_perceptions.values():
        all_percs.update(v)
    print(f"  Unique perceptions reached: {len(all_percs)}")

    if not all_percs:
        print("  SKIP: no perceptions reachable — profile analysis not possible")
        return {}, {}

    # Build perception × agent_type matrix
    per_type = {}
    for agent_n, per_set in agent_perceptions.items():
        atype = str(G.nodes[agent_n].get("type", "unknown"))
        for p in per_set:
            per_type.setdefault(p, {})
            per_type[p][atype] = per_type[p].get(atype, 0) + 1

    # Profile perceptions by dominant agent type
    profile_counts = {}
    for p, type_counts in per_type.items():
        total = sum(type_counts.values())
        dominant = max(type_counts, key=type_counts.get)
        dom_pct = type_counts[dominant] / total
        if dom_pct >= 0.6:
            profile = f"{dominant}_dominated"
        elif dom_pct >= 0.3:
            profile = f"mixed_{dominant}"
        else:
            profile = "balanced"
        profile_counts[profile] = profile_counts.get(profile, 0) + 1

    print(f"\n  Perception profiles by agent type neighbourhood:")
    for p, c in sorted(profile_counts.items()):
        print(f"    {p}: {c} perceptions")

    records = []
    for p, type_counts in per_type.items():
        label_val = node_label(G, p) if G.has_node(p) else p
        total = sum(type_counts.values())
        row = {"perception_global_id": p, "perception_label": str(label_val)[:60]}
        for atype in sorted(set(list(agents_by_type.keys()) + list(type_counts.keys()))):
            row[f"{atype}_count"] = type_counts.get(atype, 0)
            row[f"{atype}_pct"] = round(type_counts.get(atype, 0) / max(total, 1), 3)
        records.append(row)
    df = pd.DataFrame(records) if records else pd.DataFrame()
    if not df.empty:
        write_frame(df, "173_neighbourhood_profiles.csv")
        print(f"  Saved: 173_neighbourhood_profiles.csv ({len(df)} perceptions)")

    return agent_perceptions, per_type


# ===================================================================
# ANALYSIS 2: Semantic Centrality (Graph-only)
# ===================================================================
#   WHY GRAPH? Betweenness, PageRank, eigenvector centrality are
#   purely topological. They require the full graph to compute
#   which quotes bridge between narrative communities.
# ===================================================================

def run_semantic_centrality(G_multi):
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Semantic Centrality (Graph-only)")
    print("Path-based topological measures impossible on flat records")
    print("=" * 60)

    G_sem = nx.Graph()
    for u, v, d in G_multi.edges(data=True):
        if d.get("edge_family") == "qualitative_narrative":
            for n in (u, v):
                if not G_sem.has_node(n):
                    attrs = dict(G_multi.nodes[n])
                    G_sem.add_node(n, **attrs)
            G_sem.add_edge(u, v, **dict(d))

    print(f"  Semantic subgraph: {G_sem.number_of_nodes()} nodes, {G_sem.number_of_edges()} edges")

    if G_sem.number_of_nodes() < 3:
        print("  SKIP: too small for meaningful centrality")
        return

    betweenness = nx.betweenness_centrality(G_sem, normalized=True)
    pagerank = nx.pagerank(G_sem, alpha=0.85)
    eig = nx.eigenvector_centrality(G_sem, max_iter=1000, tol=1e-4)
    degree_c = nx.degree_centrality(G_sem)
    closeness = nx.closeness_centrality(G_sem)

    records = []
    for n in G_sem.nodes():
        label_val = node_label(G_sem, n)
        ntype = G_sem.nodes[n].get("node_type", "unknown")
        records.append({
            "global_id": n,
            "label": str(label_val)[:60],
            "node_type": ntype,
            "betweenness_centrality": round(betweenness.get(n, 0), 6),
            "pagerank": round(pagerank.get(n, 0), 6),
            "eigenvector_centrality": round(eig.get(n, 0), 6),
            "degree_centrality": round(degree_c.get(n, 0), 6),
            "closeness_centrality": round(closeness.get(n, 0), 6),
        })

    df = pd.DataFrame(records)
    write_frame(df, "173_semantic_centrality.csv")
    print(f"  Saved: 173_semantic_centrality.csv ({len(df)} nodes)")

    top_b = df.nlargest(10, "betweenness_centrality")
    print(f"\n  Top 10 semantic bridges (betweenness centrality):")
    print(f"  {'Label':<55} {'Type':<15} {'Betweenness':<12} {'PageRank':<10}")
    print(f"  {'-'*55} {'-'*15} {'-'*12} {'-'*10}")
    for _, r in top_b.iterrows():
        print(f"  {r['label'][:53]:<55} {r['node_type']:<15} {r['betweenness_centrality']:<12.4f} {r['pagerank']:<10.4f}")

    # Plot top 20
    top20 = df.nlargest(20, "betweenness_centrality")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors_bar = ["#e63946" if t == "information" else "#457b9d" for t in top20["node_type"]]
    ax.barh(range(len(top20)), top20["betweenness_centrality"], color=colors_bar, height=0.6)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([f"{r['label'][:35]} ({r['node_type'][:6]})" for _, r in top20.iterrows()], fontsize=8)
    ax.set_xlabel("Betweenness centrality", fontsize=11)
    ax.set_title("Top 20 Semantic Bridges — Platform 173 (Ireland)\n(graph-only: path-based topological measure)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "semantic_betweenness_top20.png", dpi=150)
    plt.close(fig)
    print(f"  Plot: semantic_betweenness_top20.png")

    top_pr = df.nlargest(10, "pagerank")
    print(f"\n  Top 10 by PageRank (influence centrality):")
    for _, r in top_pr.iterrows():
        print(f"    {r['label'][:50]:50s} PR={r['pagerank']:.4f} type={r['node_type']}")

    return df


# ===================================================================
# ANALYSIS 3: Narrative Diffusion by Agent TYPE (Graph Walk)
# ===================================================================
#   WHY GRAPH? Personalized PageRank from different agent-type
#   seeds reveals structural narrative segregation: do NGO agents
#   occupy a different part of the narrative space than
#   public_administration agents? Only a graph can compute this
#   because it requires random walks across heterogeneous edges.
# ===================================================================

def run_narrative_diffusion(G_multi):
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Narrative Diffusion by Agent Type")
    print("Graph-only: personalized PageRank across heterogeneous edges")
    print("=" * 60)

    G = simple_graph(G_multi)

    # Choose the two largest agent types for contrast
    type_groups = {}
    for n, d in G.nodes(data=True):
        if d.get("node_type") != "agent":
            continue
        atype = d.get("type", "unknown")
        if pd.isna(atype) or str(atype).strip() == "":
            atype = "unknown"
        type_groups.setdefault(str(atype), []).append(n)

    # Sort by size and pick top 2 distinct types
    sorted_types = sorted(type_groups.items(), key=lambda x: -len(x[1]))
    if len(sorted_types) < 2:
        print("  SKIP: need at least 2 agent types")
        return

    type_a_name, type_a_nodes = sorted_types[0]
    type_b_name, type_b_nodes = sorted_types[1]
    print(f"  Contrasting: {type_a_name} ({len(type_a_nodes)} agents) vs {type_b_name} ({len(type_b_nodes)} agents)")

    # Personalized PageRank
    personal_a = {n: 1.0/len(type_a_nodes) for n in type_a_nodes}
    personal_b = {n: 1.0/len(type_b_nodes) for n in type_b_nodes}

    if G.has_node(list(personal_a.keys())[0]):
        pr_a = nx.pagerank(G, personalization=personal_a, alpha=0.85, max_iter=200)
    else:
        pr_a = {}
    if G.has_node(list(personal_b.keys())[0]):
        pr_b = nx.pagerank(G, personalization=personal_b, alpha=0.85, max_iter=200)
    else:
        pr_b = {}

    records = []
    all_nodes = set(list(pr_a.keys()) + list(pr_b.keys()))
    for n in all_nodes:
        a = pr_a.get(n, 0)
        b = pr_b.get(n, 0)
        diff = (a - b) / max(a + b, 1e-10)
        ntype = G.nodes[n].get("node_type", "") if G.has_node(n) else ""
        label_val = node_label(G, n) if G.has_node(n) else n
        atype = G.nodes[n].get("type", "") if G.has_node(n) and ntype == "agent" else ""
        records.append({
            "global_id": n,
            "label": str(label_val)[:60],
            "node_type": ntype,
            "agent_type": atype,
            f"pr_{type_a_name}": round(a, 8),
            f"pr_{type_b_name}": round(b, 8),
            "diffusion_bias": round(diff, 4),
        })

    df = pd.DataFrame(records).sort_values("diffusion_bias", ascending=False)
    write_frame(df, "173_narrative_diffusion.csv")
    print(f"  Saved: 173_narrative_diffusion.csv ({len(df)} nodes)")

    bias_col = "diffusion_bias"
    # Positive = biased toward type_a
    top_a = df[df[bias_col] > 0.3].nlargest(10, bias_col)
    top_b = df[df[bias_col] < -0.3].nsmallest(10, bias_col)

    print(f"\n  Nodes most reached by {type_a_name} walk (positive bias):")
    for _, r in top_a.iterrows():
        print(f"    {r['label'][:50]:50s} bias={r[bias_col]:.3f} type={r['node_type']}")
    print(f"\n  Nodes most reached by {type_b_name} walk (negative bias):")
    for _, r in top_b.iterrows():
        print(f"    {r['label'][:50]:50s} bias={r[bias_col]:.3f} type={r['node_type']}")

    print(f"\n  Mean diffusion bias by node type:")
    grp = df.groupby("node_type")[bias_col].describe()
    print(grp.to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for ntype in ["information", "project", "agent", "channel", "challenge", "perception"]:
        subset = df[df["node_type"] == ntype]
        if not subset.empty:
            ax.hist(subset[bias_col], bins=20, alpha=0.5, label=ntype)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel(f"Diffusion bias (positive = {type_a_name} neighbourhood)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Graph Diffusion: {type_a_name} vs {type_b_name} Narrative Space\n(Platform 173, Ireland)\n(graph-only: random walk across heterogeneous edges)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "narrative_diffusion_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Plot: narrative_diffusion_distribution.png")

    return df


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 60)
    print("GRAPH-ONLY ANALYSES — Platform 173 (Ireland)")
    print("Three analyses impossible on flat record-based structures")
    print("=" * 60)

    G_multi = build_graph(directed=False)

    neighbourhood_results = run_agent_neighbourhood_profiling(G_multi)
    centrality_df = run_semantic_centrality(G_multi)
    diffusion_df = run_narrative_diffusion(G_multi)

    summary = {
        "platform_id": "173",
        "analysis_1": "Agent neighbourhood profiling by type — multi-hop BFS through agent→initiative→perception",
        "analysis_2": "Semantic centrality — betweenness, PageRank, eigenvector on qualitative_narrative subgraph",
        "analysis_3": f"Narrative diffusion — personalized PageRank contrasting top-2 agent types",
        "why_graph": {
            "analysis_1": "BFS multi-hop traversal through heterogeneous node types. Flat tables store only direct relations; graph captures topological 'reach'.",
            "analysis_2": "Betweenness/PageRank/eigenvector are path-based topological measures requiring full connectivity structure.",
            "analysis_3": "Personalized PageRank propagates agent-type signals across heterogeneous edges — measures structural narrative segregation undefined outside a graph.",
        },
    }
    if neighbourhood_results and isinstance(neighbourhood_results, tuple) and len(neighbourhood_results) > 1:
        summary["perceptions_profiled"] = len(neighbourhood_results[1])
    if centrality_df is not None and not centrality_df.empty:
        summary["semantic_nodes_analyzed"] = len(centrality_df)
    if diffusion_df is not None and not diffusion_df.empty:
        summary["diffusion_nodes_scored"] = len(diffusion_df)

    write_json(ANALYSIS_DIR / "graph_only_analyses_summary.json", summary)

    print("\n" + "=" * 60)
    print("GRAPH-ONLY ANALYSES COMPLETE — Platform 173")
    print("=" * 60)
    print(f"\nPlots: {PLOTS_DIR}")
    for f in sorted(PLOTS_DIR.iterdir()):
        print(f"  {f.name}")
    print(f"\nReports: {ANALYSIS_DIR}")
    print("  173_neighbourhood_profiles.csv")
    print("  173_semantic_centrality.csv")
    print("  173_narrative_diffusion.csv")


if __name__ == "__main__":
    main()

"""Graph-only analyses for Platform 10 (Chile COPOLAD).
Three analyses that REQUIRE graph structure — impossible on flat records.

Why graph structure?
  - Multi-hop traversal: trace agent → initiative → perception → semantic cluster
  - Topological centrality: betweenness, PageRank, eigenvector (path-based metrics)
  - Diffusion: propagate investment signals across heterogeneous edge types

Usage:
  set KTOOL_PLATFORM_ID=10 & set KTOOL_OUTPUT_SUBDIR=test
  python src/analysis/graph_only_analyses_p10.py
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

COLORS = {"high":"#e63946","medium":"#2a9d8f","low":"#f4a261","unknown":"#cccccc"}


def node_label(G, n, maxlen=60):
    """Extract best display label for a node."""
    d = G.nodes[n]
    ntype = d.get("node_type", "")
    if ntype == "information":
        txt = d.get("quote", d.get("description", ""))
        return str(txt)[:maxlen]
    return str(d.get("label", d.get("name", d.get("title", n))))[:maxlen]

# ===================================================================
# ANALYSIS 1: Agent Neighbourhood Profiling (Graph Multi-hop)
# ===================================================================
#   WHY GRAPH? A flat table joins agent→initiative→perception as a
#   single record. But the GRAPH lets us compute which perceptual
#   communities an agent reaches via ALL paths (including semantic
#   edges that connect information nodes to each other). This is
#   a neighbourhood-overlap measure that requires graph traversal.
#
#   Method: Project agent investment labels onto perception nodes
#   via weighted paths. Then cluster perceptions by shared agent
#   investment profiles. Graph-only because the "perception
#   neighbourhood" of an agent is defined by the set of all nodes
#   reachable within 2-3 hops — a topological not tabular concept.
# ===================================================================

HOP_CAP = 3

def run_agent_neighbourhood_profiling(G_multi):
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Agent Neighbourhood Profiling")
    print("Graph-only: multi-hop traversal through heterogeneous edges")
    print("=" * 60)

    G = simple_graph(G_multi)

    # Find all agent nodes with investment labels
    agents = [(n, d) for n, d in G.nodes(data=True)
              if d.get("node_type") == "agent" and d.get("investment", "") in ("high","medium","low")]

    print(f"  Labelled agents: {len(agents)}")

    # For each agent, find the set of perception nodes within HOP_CAP hops
    # that does NOT pass through another agent (we want the agent's unique footprint)
    agent_perceptions = {}
    for n, d in agents:
        # BFS-limited traversal skipping other agents
        visited = {n}
        frontier = {n}
        perceptions = set()
        for _ in range(HOP_CAP):
            next_frontier = set()
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
                    next_frontier.add(nb)
            frontier = next_frontier
        agent_perceptions[n] = perceptions

    print(f"  Agents with reachable perceptions: {sum(1 for v in agent_perceptions.values() if v)}")

    # Build a perception × investment matrix: for each perception node,
    # what fraction of agents reaching it are high/medium/low?
    per_inv = {}
    for agent_n, per_set in agent_perceptions.items():
        inv = G.nodes[agent_n].get("investment", "unknown")
        for p in per_set:
            per_inv.setdefault(p, {"high":0,"medium":0,"low":0,"total":0})
            per_inv[p][inv] += 1
            per_inv[p]["total"] += 1

    print(f"  Perceptions reached by at least one agent: {len(per_inv)}")

    # Summarize: which perception profiles exist?
    profile_counts = {}
    for p, counts in per_inv.items():
        high_pct = counts["high"] / max(counts["total"], 1)
        low_pct = counts["low"] / max(counts["total"], 1)
        if high_pct >= 0.6:
            profile = "high_dominated"
        elif low_pct >= 0.6:
            profile = "low_dominated"
        elif high_pct >= 0.3:
            profile = "mixed_high"
        elif low_pct >= 0.3:
            profile = "mixed_low"
        else:
            profile = "balanced"
        profile_counts[profile] = profile_counts.get(profile, 0) + 1

    print(f"\n  Perception investment profiles (graph-determined):")
    for p, c in sorted(profile_counts.items()):
        print(f"    {p}: {c} perceptions")

    # Persist
    records = []
    for p, counts in per_inv.items():
        label = node_label(G, p)
        total = counts["total"]
        records.append({
            "perception_global_id": p,
            "perception_label": str(label)[:60],
            "total_reaching_agents": total,
            "high_pct": round(counts["high"]/max(total,1), 3),
            "medium_pct": round(counts["medium"]/max(total,1), 3),
            "low_pct": round(counts["low"]/max(total,1), 3),
        })
    df = pd.DataFrame(records).sort_values("high_pct", ascending=False)
    write_frame(df, "p10_neighbourhood_profiles.csv")
    print(f"  Saved: p10_neighbourhood_profiles.csv ({len(df)} perceptions)")

    # Plot: perception profile distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_plot = {"high_dominated":"#e63946","low_dominated":"#f4a261",
                   "mixed_high":"#d9788f","mixed_low":"#e8b88a","balanced":"#457b9d"}
    bars = ax.bar(list(profile_counts.keys()), list(profile_counts.values()),
                  color=[colors_plot.get(k,"#ccc") for k in profile_counts.keys()])
    for bar, val in zip(bars, profile_counts.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                str(val), ha="center", fontweight="bold")
    ax.set_xlabel("Investment profile (graph-derived perception neighbourhood)", fontsize=11)
    ax.set_ylabel("Number of perception nodes", fontsize=11)
    ax.set_title("Perception Nodes Grouped by Agent Investment Neighbourhood\n(Graph multi-hop traversal)", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "perception_investment_profiles.png", dpi=150)
    plt.close(fig)
    print(f"  Plot: perception_investment_profiles.png")

    return agent_perceptions, per_inv


# ===================================================================
# ANALYSIS 2: Semantic Centrality (Graph-only)
# ===================================================================
#   WHY GRAPH? Betweenness, PageRank, and eigenvector centrality
#   are purely topological measures. They require the full graph
#   connectivity to compute path-based influence. A flat table of
#   quotes with metadata cannot tell you which quote bridges
#   between different thematic communities — only the graph can
#   because it embeds the connectivity structure.
#
#   Method: Build a subgraph of only qualitative_narrative edges
#   (semantic). Compute four centrality measures. Rank nodes by
#   each measure. Graph-only because centrality is defined by
#   position in the network topology.
# ===================================================================

def run_semantic_centrality(G_multi):
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Semantic Centrality (Graph-only)")
    print("Path-based topological measures impossible on flat records")
    print("=" * 60)

    # Extract semantic edge subgraph
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

    # Centrality measures — ALL graph-only
    betweenness = nx.betweenness_centrality(G_sem, normalized=True)
    pagerank = nx.pagerank(G_sem, alpha=0.85)
    eig = nx.eigenvector_centrality(G_sem, max_iter=1000, tol=1e-4)
    degree_c = nx.degree_centrality(G_sem)
    closeness = nx.closeness_centrality(G_sem)

    # Build records
    records = []
    for n in G_sem.nodes():
        label = node_label(G_sem, n)
        ntype = G_sem.nodes[n].get("node_type", "unknown")
        records.append({
            "global_id": n,
            "label": str(label)[:60],
            "node_type": ntype,
            "betweenness_centrality": round(betweenness.get(n, 0), 6),
            "pagerank": round(pagerank.get(n, 0), 6),
            "eigenvector_centrality": round(eig.get(n, 0), 6),
            "degree_centrality": round(degree_c.get(n, 0), 6),
            "closeness_centrality": round(closeness.get(n, 0), 6),
        })

    df = pd.DataFrame(records)
    write_frame(df, "p10_semantic_centrality.csv")
    print(f"  Saved: p10_semantic_centrality.csv ({len(df)} nodes)")

    # Show top 10 by betweenness (the narrative bridges)
    top_betweenness = df.nlargest(10, "betweenness_centrality")
    print(f"\n  Top 10 semantic bridges (betweenness centrality):")
    print(f"  {'Node':<12} {'Label':<40} {'Type':<15} {'Betweenness':<12} {'PageRank':<10} {'Eigen':<10}")
    print(f"  {'-'*12} {'-'*40} {'-'*15} {'-'*12} {'-'*10} {'-'*10}")
    for _, r in top_betweenness.iterrows():
        lbl = r["label"][:38]
        print(f"  {r['global_id'][-10:]:<12} {lbl:<40} {r['node_type']:<15} {r['betweenness_centrality']:<12.4f} {r['pagerank']:<10.4f} {r['eigenvector_centrality']:<10.4f}")

    # Plot: top 20 betweenness
    top20 = df.nlargest(20, "betweenness_centrality")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors_bar = ["#e63946" if t == "information" else "#457b9d" for t in top20["node_type"]]
    bars = ax.barh(range(len(top20)), top20["betweenness_centrality"], color=colors_bar, height=0.6)
    ax.set_yticks(range(len(top20)))
    labels = [f"{r['label'][:35]} ({r['node_type'][:6]})" for _, r in top20.iterrows()]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Betweenness centrality", fontsize=11)
    ax.set_title("Top 20 Semantic Bridges in Chile COPOLAD Network\n(graph-only — path-based topological measure)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "semantic_betweenness_top20.png", dpi=150)
    plt.close(fig)
    print(f"  Plot: semantic_betweenness_top20.png")

    # PageRank top 10
    top_pr = df.nlargest(10, "pagerank")
    print(f"\n  Top 10 by PageRank (influence centrality):")
    for _, r in top_pr.iterrows():
        print(f"    {r['label'][:45]:45s}  PR={r['pagerank']:.4f}  type={r['node_type']}")

    return df


# ===================================================================
# ANALYSIS 3: Narrative-Investment Diffusion (Graph Walk)
# ===================================================================
#   WHY GRAPH? This analysis propagates investment signals through
#   the graph using personalized PageRank — a random walk that
#   starts from investment-labelled agents, walks across
#   agent→initiative→perception→information→semantic edges, and
#   measures which semantic neighborhoods are preferentially
#   reached by high-investment vs low-investment seeds.
#
#   A flat table can only store direct agent→initiative links. It
#   cannot simulate how "narrative influence" flows across
#   heterogeneous edge types to distant nodes. Graph diffusion
#   captures this: it's the only way to measure whether
#   high-investment agents inhabit a structurally different part
#   of the narrative space than low-investment agents.
# ===================================================================

def run_narrative_diffusion(G_multi):
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Narrative-Investment Diffusion")
    print("Graph-only: personalized PageRank across heterogeneous edges")
    print("=" * 60)

    G = simple_graph(G_multi)

    # Get labelled agents
    high_agents = [n for n, d in G.nodes(data=True)
                   if d.get("node_type") == "agent" and d.get("investment") == "high"]
    low_agents = [n for n, d in G.nodes(data=True)
                  if d.get("node_type") == "agent" and d.get("investment") == "low"]
    med_agents = [n for n, d in G.nodes(data=True)
                  if d.get("node_type") == "agent" and d.get("investment") == "medium"]

    print(f"  Seed agents: high={len(high_agents)}, medium={len(med_agents)}, low={len(low_agents)}")

    if not (high_agents and low_agents):
        print("  SKIP: need both high and low agents")
        return

    # Personalized PageRank from high-investment seeds
    personal_high = {n: 1.0/len(high_agents) for n in high_agents} if high_agents else None
    personal_low = {n: 1.0/len(low_agents) for n in low_agents} if low_agents else None

    if personal_high and G.has_node(list(personal_high.keys())[0]):
        pr_high = nx.pagerank(G, personalization=personal_high, alpha=0.85, max_iter=200)
    else:
        pr_high = {}
    if personal_low and G.has_node(list(personal_low.keys())[0]):
        pr_low = nx.pagerank(G, personalization=personal_low, alpha=0.85, max_iter=200)
    else:
        pr_low = {}

    # Compute the diffusion DIFFERENCE: which nodes are more "high-investment" in their narrative neighbourhood?
    records = []
    all_nodes = set(list(pr_high.keys()) + list(pr_low.keys()))
    for n in all_nodes:
        h = pr_high.get(n, 0)
        l = pr_low.get(n, 0)
        # Normalized difference
        diff = (h - l) / max(h + l, 1e-10)
        ntype = G.nodes[n].get("node_type", "") if G.has_node(n) else ""
        label = node_label(G, n) if G.has_node(n) else n
        inv = G.nodes[n].get("investment", "") if G.has_node(n) else ""
        records.append({
            "global_id": n,
            "label": str(label)[:60],
            "node_type": ntype,
            "investment": inv if ntype == "agent" else "",
            "pr_high": round(h, 8),
            "pr_low": round(l, 8),
            "diffusion_bias": round(diff, 4),
        })

    df = pd.DataFrame(records).sort_values("diffusion_bias", ascending=False)
    write_frame(df, "p10_narrative_diffusion.csv")
    print(f"  Saved: p10_narrative_diffusion.csv ({len(df)} nodes)")

    # Show nodes most biased toward high-investment narratives
    top_high = df[df["diffusion_bias"] > 0.3].nlargest(10, "diffusion_bias")
    top_low = df[df["diffusion_bias"] < -0.3].nsmallest(10, "diffusion_bias")
    print(f"\n  Nodes most reached by HIGH-investment walk:")
    for _, r in top_high.iterrows():
        print(f"    {r['label'][:50]:50s} bias={r['diffusion_bias']:.3f} type={r['node_type']}")
    print(f"\n  Nodes most reached by LOW-investment walk:")
    for _, r in top_low.iterrows():
        print(f"    {r['label'][:50]:50s} bias={r['diffusion_bias']:.3f} type={r['node_type']}")

    # Distribution of diffusion bias by node type
    print(f"\n  Mean diffusion bias by node type:")
    print(df.groupby("node_type")["diffusion_bias"].describe().to_string())

    # Plot: diffusion bias distribution by node type
    fig, ax = plt.subplots(figsize=(10, 6))
    types = df[df["node_type"].isin(["information","perception","challenge","theme","agent"])]
    for ntype in ["agent","information","perception","challenge","theme"]:
        subset = types[types["node_type"] == ntype]
        if not subset.empty:
            ax.hist(subset["diffusion_bias"], bins=30, alpha=0.5, label=ntype)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Diffusion bias (positive = high-investment narrative neighbourhood)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Graph Diffusion: Which Parts of the Narrative Space Do\nHigh vs Low Investment Agents Occupy?\n(graph-only — random walk across heterogeneous edges)", fontsize=11, fontweight="bold")
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
    print("GRAPH-ONLY ANALYSES — Platform 10 (Chile COPOLAD)")
    print("Three analyses impossible on flat record-based structures")
    print("=" * 60)

    G_multi = build_graph(directed=False)

    # 1
    agent_perceptions, per_inv = run_agent_neighbourhood_profiling(G_multi)

    # 2
    centrality_df = run_semantic_centrality(G_multi)

    # 3
    diffusion_df = run_narrative_diffusion(G_multi)

    # Compile summary
    summary = {
        "platform_id": "10",
        "description": "Graph-only analyses — Platform 10 Chile COPOLAD",
        "why_graph": {
            "analysis_1": "Agent neighbourhood profiling requires BFS multi-hop traversal through heterogeneous node types (agent→initiative→perception→semantic edge). A flat table can only store direct relationships; the graph captures the topological 'reach' of each agent in narrative space.",
            "analysis_2": "Semantic centrality (betweenness, PageRank, eigenvector) are purely path-based topological measures. They require the full connectivity structure to compute which nodes bridge between communities. No tabular structure can compute betweenness because it depends on shortest paths through the entire graph.",
            "analysis_3": "Narrative-investment diffusion uses personalized PageRank — a random walk that propagates investment signals across heterogeneous edge types (declared_relational → interpretive → qualitative_narrative). This measures structural affinity between investment classes and narrative positions, which is undefined outside a graph.",
        },
        "analysis_1_summary": {
            "labelled_agents": len(agent_perceptions),
            "perceptions_with_profiles": len(per_inv),
        },
        "analysis_2_summary": {
            "semantic_nodes_analyzed": len(centrality_df) if centrality_df is not None and not centrality_df.empty else 0,
        },
        "analysis_3_summary": {
            "nodes_with_diffusion_score": len(diffusion_df) if diffusion_df is not None and not diffusion_df.empty else 0,
        },
    }
    write_json(ANALYSIS_DIR / "graph_only_analyses_summary.json", summary)

    print("\n" + "=" * 60)
    print("GRAPH-ONLY ANALYSES COMPLETE")
    print("=" * 60)
    print(f"\nPlots: {PLOTS_DIR}")
    for f in sorted(PLOTS_DIR.iterdir()):
        print(f"  {f.name}")
    print(f"\nReports: {ANALYSIS_DIR}")
    print("  p10_neighbourhood_profiles.csv")
    print("  p10_semantic_centrality.csv")
    print("  p10_narrative_diffusion.csv")
    print("  graph_only_analyses_summary.json")


if __name__ == "__main__":
    main()

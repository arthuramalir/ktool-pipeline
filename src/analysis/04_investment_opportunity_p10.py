"""Investment analysis for Platform 10 (Chile COPOLAD).
Focus: investment levels (high/medium/low) labelled on agents.

Extends the standard 04_investment_opportunity.py with:
  - Investment-level distribution across agents
  - Agent type vs investment cross-tab
  - People-involved vs investment analysis
  - Thematic area concentration by investment level
  - Bridge scores by investment tier

Usage:
    set KTOOL_PLATFORM_ID=10 & set KTOOL_OUTPUT_SUBDIR=test
    python src/analysis/04_investment_opportunity_p10.py
"""

from __future__ import annotations
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_utils import (
    ANALYSIS_DIR, ANALYTICS_DIR, DATA_DIR, PLATFORM_ID,
    build_graph, simple_graph, enrich_with_node_metadata,
    write_frame, write_json, load_nodes_edges,
)

PLOTS_DIR = ANALYSIS_DIR / "investment_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    pass


def plot_investment_distribution(agents_df, save_path):
    """Bar chart of agent count by investment level."""
    counts = agents_df["investment"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"low": "#f4a261", "medium": "#2a9d8f", "high": "#e63946"}
    bars = ax.bar(counts.index, counts.values,
                  color=[colors.get(k, "#457b9d") for k in counts.index],
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha="center", fontsize=12, fontweight="bold")
    ax.set_xlabel("Investment level", fontsize=12)
    ax.set_ylabel("Number of agents", fontsize=12)
    ax.set_title("Agent Investment Distribution\nChile COPOLAD Platform", fontsize=14, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  Plot: {save_path.name}")


def plot_investment_by_type(agents_df, save_path):
    """Stacked bar: agent type × investment level."""
    ct = pd.crosstab(agents_df["type"].fillna("unknown"), agents_df["investment"].fillna("unknown"))
    # Order by total count
    ct["total"] = ct.sum(axis=1)
    ct = ct.sort_values("total", ascending=True).drop(columns="total")
    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(kind="barh", stacked=True, ax=ax,
            color=["#f4a261", "#2a9d8f", "#e63946"],
            edgecolor="white", width=0.7)
    ax.set_xlabel("Number of agents", fontsize=12)
    ax.set_ylabel("Agent type", fontsize=11)
    ax.set_title("Investment Level by Agent Type", fontsize=14, fontweight="bold")
    ax.legend(title="Investment", loc="lower right")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  Plot: {save_path.name}")


def plot_investment_vs_people(agents_df, save_path):
    """Grouped bar: people_involved × investment level."""
    ct = pd.crosstab(agents_df["people_involved"].fillna("unknown"),
                     agents_df["investment"].fillna("unknown"))
    order = ["from_1_to_10", "from_10_to_50", "more_than_50", "more_than_100"]
    ct = ct.reindex([o for o in order if o in ct.index])
    fig, ax = plt.subplots(figsize=(9, 5))
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=["#f4a261", "#2a9d8f", "#e63946"],
            edgecolor="white", width=0.6)
    ax.set_xlabel("People involved", fontsize=12)
    ax.set_ylabel("Number of agents", fontsize=12)
    ax.set_title("Investment Level by Organisation Size", fontsize=14, fontweight="bold")
    ax.legend(title="Investment")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  Plot: {save_path.name}")


def plot_investment_bridge_scores(graph, ranked_nodes, ranked_agents, save_path, betweenness_dict=None):
    """Scatter: bridge score (betweenness) by investment tier for agents."""
    if betweenness_dict is None:
        betweenness_dict = nx.betweenness_centrality(graph)
    records = []
    for n in graph.nodes():
        if graph.nodes[n].get("node_type") == "agent":
            inv = graph.nodes[n].get("investment", "unknown")
            records.append({
                "label": graph.nodes[n].get("label", n)[:30],
                "node_id": n,
                "investment": inv,
                "betweenness": betweenness_dict.get(n, 0),
                "degree": graph.degree(n),
            })
    df = pd.DataFrame(records)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"low": "#f4a261", "medium": "#2a9d8f", "high": "#e63946", "unknown": "#cccccc"}
    for inv in ["high", "medium", "low", "unknown"]:
        subset = df[df["investment"] == inv]
        if not subset.empty:
            ax.scatter(subset["degree"], subset["betweenness"],
                       c=colors.get(inv, "#cccccc"), label=f"{inv} investment",
                       s=80, alpha=0.7, edgecolors="white", linewidth=0.5)
    # Label top agents
    for _, r in df.nlargest(8, "betweenness").iterrows():
        ax.annotate(r["label"], (r["degree"], r["betweenness"]),
                    fontsize=7, alpha=0.8, ha="left")
    ax.set_xlabel("Degree (local connections)", fontsize=12)
    ax.set_ylabel("Betweenness centrality (bridge score)", fontsize=12)
    ax.set_title("Bridge Score vs Local Connectivity by Investment Level", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  Plot: {save_path.name}")


def run():
    print("=" * 60)
    print(f"INVESTMENT ANALYSIS — Platform {PLATFORM_ID} (Chile COPOLAD)")
    print("=" * 60)

    # Load graph
    G_multi = build_graph(directed=False)
    G = simple_graph(G_multi)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load nodes for agent data
    nodes_df, edges_df = load_nodes_edges()

    # --- Agent Investment Summary ---
    agents = nodes_df[nodes_df["node_type"] == "agent"].copy()
    print(f"\nAgents with investment data: {agents['investment'].notna().sum()} / {len(agents)}")
    print(f"\nInvestment distribution:")
    print(agents["investment"].value_counts().to_string())

    if agents.empty:
        print("ERROR: No agent nodes found")
        return

    # --- Investment distribution plot ---
    plot_investment_distribution(agents, PLOTS_DIR / "investment_distribution.png")
    plot_investment_by_type(agents, PLOTS_DIR / "investment_by_agent_type.png")
    plot_investment_vs_people(agents, PLOTS_DIR / "investment_vs_people_involved.png")

    # --- Agent type summary ---
    print(f"\nAgent type distribution:")
    print(agents["type"].value_counts().to_string())

    # --- Investment × Type cross-tab ---
    ct = pd.crosstab(agents["type"].fillna("unknown"), agents["investment"].fillna("unknown"))
    print(f"\nInvestment × Agent type:")
    print(ct.to_string())

    # --- People involved × Investment ---
    ct2 = pd.crosstab(agents["people_involved"].fillna("unknown"), agents["investment"].fillna("unknown"))
    print(f"\nInvestment × People involved:")
    print(ct2.to_string())

    # --- Bridge scores by investment ---
    betweenness = nx.betweenness_centrality(G)
    degree_cent = nx.degree_centrality(G)

    agent_metrics = []
    for n in G.nodes():
        if G.nodes[n].get("node_type") == "agent":
            agent_metrics.append({
                "global_id": n,
                "label": G.nodes[n].get("label", n),
                "agent_type": G.nodes[n].get("type", ""),
                "investment": G.nodes[n].get("investment", ""),
                "people_involved": G.nodes[n].get("people_involved", ""),
                "betweenness_centrality": round(betweenness.get(n, 0), 6),
                "degree_centrality": round(degree_cent.get(n, 0), 6),
                "degree": G.degree(n),
            })
    df_metrics = pd.DataFrame(agent_metrics).sort_values("betweenness_centrality", ascending=False)
    write_frame(df_metrics, "p10_agent_investment_metrics.csv")
    print(f"\nTop 10 agents by bridge score:")
    print(df_metrics[["label", "investment", "agent_type", "betweenness_centrality", "degree"]].head(10).to_string(index=False))

    # --- Investment × bridge score summary ---
    if not df_metrics.empty:
        print(f"\nMean bridge score by investment level:")
        print(df_metrics.groupby("investment")["betweenness_centrality"].describe().to_string())

    plot_investment_bridge_scores(G, [], [], PLOTS_DIR / "investment_vs_bridge_score.png", betweenness_dict=betweenness)

    # --- Challenge attention (from perception_challenge_links) ---
    challenge_attention = {}
    for u, v, d in G_multi.edges(data=True):
        if d.get("edge_type") == "perception_reveals_challenge":
            ch_id = v
            challenge_attention[ch_id] = challenge_attention.get(ch_id, 0) + 1

    challenge_records = []
    for cid, count in challenge_attention.items():
        label = G.nodes[cid].get("label", cid) if G.has_node(cid) else cid
        challenge_records.append({"global_id": cid, "label": label, "perception_count_weight": count})
    df_challenge = pd.DataFrame(challenge_records).sort_values("perception_count_weight", ascending=False)
    if not df_challenge.empty:
        write_frame(df_challenge, "p10_challenge_attention.csv")
        print(f"\nTop challenges by attention:")
        print(df_challenge.head(10).to_string(index=False))

    # --- Comprehensive report ---
    report = {
        "platform_id": PLATFORM_ID,
        "platform_name": "Chile COPOLAD",
        "total_agents": len(agents),
        "agents_with_investment": int(agents["investment"].notna().sum()),
        "investment_distribution": agents["investment"].value_counts().to_dict(),
        "agent_type_distribution": agents["type"].value_counts().to_dict(),
        "top_bridge_agents": df_metrics.head(10)[["label", "investment", "betweenness_centrality"]].to_dict("records"),
        "total_graph_nodes": G.number_of_nodes(),
        "total_graph_edges": G.number_of_edges(),
    }
    write_json(ANALYSIS_DIR / "p10_investment_summary.json", report)

    print(f"\n{'='*60}")
    print("INVESTMENT ANALYSIS COMPLETE")
    print(f"Plots: {PLOTS_DIR}")
    print(f"Reports: {ANALYSIS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()

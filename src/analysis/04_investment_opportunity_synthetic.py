"""Financial Simulation — Platform 173_synthetic.

Runs investment-weighted graph analyses on the synthetic 173 dataset which
has deterministic budget and investment-level columns.

Three analyses:
  A. Value Leverage Matrix  — betweenness centrality / budget (€M)
  B. Stranded Asset Detection — high budget + zero bridge score + low degree
  C. Financial-Weighted Personalized PageRank — investment_eur_estimate
     as seed weights; reveals which narrative space financial capital inhabits

Usage:
  set KTOOL_PLATFORM_ID=173_synthetic & set KTOOL_OUTPUT_SUBDIR=test
  python src/analysis/04_investment_opportunity_synthetic.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_utils import (
    ANALYSIS_DIR, ANALYTICS_DIR, DATA_DIR, PLATFORM_ID,
    build_graph, simple_graph, load_nodes_edges, write_frame, write_json,
)

PLOTS_DIR = ANALYSIS_DIR / "financial_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _node_label(G: nx.Graph, n: str, maxlen: int = 50) -> str:
    d = G.nodes[n]
    return str(d.get("label", d.get("name", d.get("title", n))))[:maxlen]


# ---------------------------------------------------------------------------
# A. Value Leverage Matrix
# ---------------------------------------------------------------------------

def run_value_leverage(G: nx.Graph, nodes_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("A. Value Leverage Matrix (betweenness / budget)")
    print("=" * 60)

    betweenness = nx.betweenness_centrality(G, normalized=True)

    records = []
    for n, d in G.nodes(data=True):
        if d.get("node_type") not in {"project", "pilot", "prototype"}:
            continue
        label = _node_label(G, n)
        budget = float(d.get("associated_budget", 0) or 0)
        bridge = betweenness.get(n, 0)
        degree = G.degree(n)
        inv_level = str(d.get("investment_level", ""))
        # Leverage: bridge score per €1M budget (avoid /0)
        leverage = bridge / max(budget / 1_000_000, 0.001)
        records.append({
            "global_id": n,
            "label": label,
            "node_type": d.get("node_type", ""),
            "associated_budget": int(budget),
            "investment_level": inv_level,
            "betweenness_centrality": round(bridge, 6),
            "degree": degree,
            "leverage_score": round(leverage, 6),
        })

    df = pd.DataFrame(records).sort_values("leverage_score", ascending=False)
    write_frame(df, "synthetic_value_leverage.csv")
    print(f"  Saved: synthetic_value_leverage.csv  ({len(df)} initiatives)")

    # Top/bottom
    top10 = df.head(10)
    print(f"\n  Top 10 HIGH-LEVERAGE (most bridge value per euro):")
    for _, r in top10.iterrows():
        print(f"    {r['label'][:40]:40s} leverage={r['leverage_score']:.4f}  budget=€{r['associated_budget']:,}  bridge={r['betweenness_centrality']:.4f}")

    bottom10 = df.tail(10)
    print(f"\n  Bottom 10 LOW-LEVERAGE (expensive, low bridge):")
    for _, r in bottom10.iterrows():
        print(f"    {r['label'][:40]:40s} leverage={r['leverage_score']:.4f}  budget=€{r['associated_budget']:,}  bridge={r['betweenness_centrality']:.4f}")

    # Scatter plot: budget vs betweenness coloured by investment_level
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_map = {"low": "#f4a261", "medium": "#2a9d8f", "high": "#e63946", "": "#aaaaaa"}
    for inv_level, grp in df.groupby("investment_level"):
        ax.scatter(
            grp["associated_budget"] / 1_000_000,
            grp["betweenness_centrality"],
            c=colors_map.get(str(inv_level).lower(), "#aaaaaa"),
            label=f"{inv_level} investment",
            s=70, alpha=0.8, edgecolors="white", linewidth=0.5,
        )
    # Annotate top 8 by leverage
    for _, r in df.head(8).iterrows():
        ax.annotate(
            r["label"][:25],
            (r["associated_budget"] / 1_000_000, r["betweenness_centrality"]),
            fontsize=7, alpha=0.85, ha="left",
        )
    ax.set_xlabel("Budget (€M)", fontsize=12)
    ax.set_ylabel("Betweenness centrality (bridge score)", fontsize=12)
    ax.set_title(
        "Value Leverage: Budget vs Network Bridge Score\n(Platform 173 Synthetic — Ireland)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "value_leverage_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Plot: value_leverage_scatter.png")

    return df


# ---------------------------------------------------------------------------
# B. Stranded Asset Detection
# ---------------------------------------------------------------------------

def run_stranded_assets(G: nx.Graph, leverage_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("B. Stranded Asset Detection")
    print("=" * 60)

    betweenness = nx.betweenness_centrality(G, normalized=True)
    BUDGET_THRESHOLD = 500_000
    BRIDGE_THRESHOLD = 0.001
    DEGREE_THRESHOLD = 3

    records = []
    for n, d in G.nodes(data=True):
        if d.get("node_type") not in {"project", "pilot", "prototype"}:
            continue
        budget = float(d.get("associated_budget", 0) or 0)
        bridge = betweenness.get(n, 0)
        degree = G.degree(n)
        if budget >= BUDGET_THRESHOLD and bridge <= BRIDGE_THRESHOLD and degree <= DEGREE_THRESHOLD:
            records.append({
                "global_id": n,
                "label": _node_label(G, n),
                "node_type": d.get("node_type", ""),
                "associated_budget": int(budget),
                "investment_level": str(d.get("investment_level", "")),
                "betweenness_centrality": round(bridge, 6),
                "degree": degree,
                "stranded_reason": f"Budget ≥ €{BUDGET_THRESHOLD:,} but bridge={bridge:.4f}, degree={degree}",
            })

    df = pd.DataFrame(records).sort_values("associated_budget", ascending=False)
    write_frame(df, "synthetic_stranded_assets.csv")
    print(f"  Found {len(df)} stranded assets (budget ≥ €{BUDGET_THRESHOLD:,}, low bridge + low degree)")
    for _, r in df.head(10).iterrows():
        print(f"    {r['label'][:40]:40s}  €{r['associated_budget']:,}  bridge={r['betweenness_centrality']:.4f}  deg={r['degree']}")

    return df


# ---------------------------------------------------------------------------
# C. Financial-Weighted Personalized PageRank
# ---------------------------------------------------------------------------

def run_financial_diffusion(G_multi: nx.Graph, G: nx.Graph) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("C. Financial-Weighted Narrative Diffusion (Personalized PageRank)")
    print("=" * 60)

    # Seed weights = investment_eur_estimate for agents
    personal: dict[str, float] = {}
    for n, d in G.nodes(data=True):
        if d.get("node_type") == "agent":
            est = float(d.get("investment_eur_estimate", 0) or 0)
            if est > 0:
                personal[n] = est

    total_invest = sum(personal.values())
    if total_invest == 0 or len(personal) < 2:
        print("  SKIP: no financial seeds found")
        return pd.DataFrame()

    # Normalise to probability distribution
    personal_norm = {n: v / total_invest for n, v in personal.items()}
    print(f"  Financial seeds: {len(personal_norm)} agents  total_invest=€{total_invest:,.0f}")

    pr_financial = nx.pagerank(G, personalization=personal_norm, alpha=0.85, max_iter=300)

    # Uniform baseline (equal seeds)
    uniform = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}
    pr_uniform = nx.pagerank(G, personalization=uniform, alpha=0.85, max_iter=300)

    records = []
    for n in G.nodes():
        f = pr_financial.get(n, 0)
        u = pr_uniform.get(n, 0)
        bias = (f - u) / max(f + u, 1e-10)
        d = G.nodes[n]
        records.append({
            "global_id": n,
            "label": _node_label(G, n),
            "node_type": d.get("node_type", ""),
            "investment_eur": float(d.get("investment_eur_estimate", 0) or 0),
            "pr_financial": round(f, 8),
            "pr_uniform": round(u, 8),
            "financial_bias": round(bias, 4),
        })

    df = pd.DataFrame(records).sort_values("financial_bias", ascending=False)
    write_frame(df, "synthetic_financial_diffusion.csv")
    print(f"  Saved: synthetic_financial_diffusion.csv  ({len(df)} nodes)")

    top_cap = df[df["financial_bias"] > 0.3].head(10)
    top_under = df[df["financial_bias"] < -0.3].tail(10)
    print(f"\n  Nodes over-reached by financial capital (bias > 0.3):")
    for _, r in top_cap.iterrows():
        print(f"    {r['label'][:45]:45s}  bias={r['financial_bias']:.3f}  type={r['node_type']}")
    print(f"\n  Nodes under-reached by financial capital (bias < -0.3):")
    for _, r in top_under.iterrows():
        print(f"    {r['label'][:45]:45s}  bias={r['financial_bias']:.3f}  type={r['node_type']}")

    # Distribution plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for ntype in ["information", "perception", "challenge", "agent", "project"]:
        sub = df[df["node_type"] == ntype]
        if not sub.empty:
            ax.hist(sub["financial_bias"], bins=25, alpha=0.5, label=ntype)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Financial bias (positive = capital-heavy narrative zone)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Financial-Weighted Narrative Diffusion\n(Platform 173 Synthetic)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "financial_diffusion_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Plot: financial_diffusion_distribution.png")

    return df


# ---------------------------------------------------------------------------
# D. Random Budget Reallocation Simulation
# ---------------------------------------------------------------------------

def run_budget_allocation_simulation(leverage_df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("D. Random Budget Reallocation Simulation")
    print("=" * 60)

    budgets = leverage_df["associated_budget"].values
    bridges = leverage_df["betweenness_centrality"].values
    n = len(budgets)
    observed_leverages = bridges / np.maximum(budgets / 1_000_000, 0.001)

    observed_mean = float(np.mean(observed_leverages))
    observed_gini = float(_gini(observed_leverages))
    observed_stranded = int(np.sum((budgets >= 500_000) & (bridges <= 0.001)))

    N_ITER = 1000
    rng = np.random.default_rng(42)
    sim_means = np.empty(N_ITER)
    sim_ginis = np.empty(N_ITER)
    sim_stranded = np.empty(N_ITER)

    for i in range(N_ITER):
        shuffled = rng.permutation(budgets)
        sim_leverages = bridges / np.maximum(shuffled / 1_000_000, 0.001)
        sim_means[i] = np.mean(sim_leverages)
        sim_ginis[i] = _gini(sim_leverages)
        sim_stranded[i] = np.sum((shuffled >= 500_000) & (bridges <= 0.001))

    mean_pct = stats.percentileofscore(sim_means, observed_mean)
    gini_pct = stats.percentileofscore(sim_ginis, observed_gini)
    stranded_pct = stats.percentileofscore(sim_stranded, observed_stranded)

    result = {
        "n_simulations": N_ITER,
        "observed_mean_leverage": round(observed_mean, 6),
        "sim_mean_leverage": round(float(np.mean(sim_means)), 6),
        "sim_std_leverage": round(float(np.std(sim_means)), 6),
        "mean_leverage_percentile": round(float(mean_pct), 1),
        "observed_gini": round(observed_gini, 4),
        "sim_mean_gini": round(float(np.mean(sim_ginis)), 4),
        "gini_percentile": round(float(gini_pct), 1),
        "observed_stranded": observed_stranded,
        "sim_mean_stranded": round(float(np.mean(sim_stranded)), 1),
        "stranded_percentile": round(float(stranded_pct), 1),
        "efficiency_verdict": "above_random" if mean_pct > 95 else (
            "below_random" if mean_pct < 5 else "indeterminate"
        ),
    }

    print(f"  Simulations: {N_ITER}")
    print(f"  Observed mean leverage: {observed_mean:.6f}  (percentile: {mean_pct:.0f}%)")
    print(f"  Simulated mean leverage: {np.mean(sim_means):.6f} ± {np.std(sim_means):.6f}")
    print(f"  Observed Gini: {observed_gini:.4f}  (percentile: {gini_pct:.0f}%)")
    print(f"  Observed stranded: {observed_stranded}  (percentile: {stranded_pct:.0f}%)")
    print(f"  Verdict: {result['efficiency_verdict']}")

    # Histogram comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist(sim_means, bins=40, color="#457b9d", alpha=0.7, edgecolor="white")
    axes[0].axvline(observed_mean, color="#e63946", linewidth=2, linestyle="--", label=f"Observed ({observed_mean:.5f})")
    axes[0].set_xlabel("Mean leverage"); axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Mean Leverage (pct={mean_pct:.0f}%)")
    axes[0].legend(fontsize=8)
    axes[1].hist(sim_ginis, bins=40, color="#2a9d8f", alpha=0.7, edgecolor="white")
    axes[1].axvline(observed_gini, color="#e63946", linewidth=2, linestyle="--", label=f"Observed ({observed_gini:.3f})")
    axes[1].set_xlabel("Gini coefficient"); axes[1].set_title(f"Leverage Inequality (pct={gini_pct:.0f}%)")
    axes[1].legend(fontsize=8)
    axes[2].hist(sim_stranded, bins=min(40, int(sim_stranded.max()) - int(sim_stranded.min()) + 1),
                 color="#e9c46a", alpha=0.7, edgecolor="white")
    axes[2].axvline(observed_stranded, color="#e63946", linewidth=2, linestyle="--", label=f"Observed ({observed_stranded})")
    axes[2].set_xlabel("Stranded asset count"); axes[2].set_title(f"Stranded Assets (pct={stranded_pct:.0f}%)")
    axes[2].legend(fontsize=8)
    fig.suptitle("Budget Allocation Simulation: Observed vs Random Reallocation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "budget_reallocation_simulation.png", dpi=150)
    plt.close(fig)
    print(f"  Plot: budget_reallocation_simulation.png")

    return result


def _gini(x: np.ndarray) -> float:
    x_sorted = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(x_sorted)
    return float((2 * np.sum((np.arange(1, n + 1) * x_sorted)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1] + 1e-10))


# ---------------------------------------------------------------------------
# Budget distribution plot
# ---------------------------------------------------------------------------

def plot_budget_distribution(nodes_df: pd.DataFrame):
    for budget_col in ("associated_budget", "budget"):
        if budget_col in nodes_df.columns:
            break
    else:
        return

    initiatives = nodes_df[nodes_df["node_type"].isin({"project", "pilot", "prototype"})].copy()
    initiatives[budget_col] = pd.to_numeric(initiatives[budget_col], errors="coerce").fillna(0)
    if initiatives.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors_map = {"project": "#2a9d8f", "pilot": "#457b9d", "prototype": "#e9c46a"}
    for ntype, grp in initiatives.groupby("node_type"):
        ax.hist(
            grp[budget_col] / 1_000,
            bins=20,
            alpha=0.6,
            label=ntype,
            color=colors_map.get(ntype, "#aaaaaa"),
        )
    ax.set_xlabel("Budget (€k)", fontsize=12)
    ax.set_ylabel("Number of initiatives", fontsize=12)
    ax.set_title("Budget Distribution by Initiative Type\n(Platform 173 Synthetic)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "budget_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Plot: budget_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(f"FINANCIAL SIMULATION — Platform {PLATFORM_ID}")
    print("=" * 60)

    G_multi = build_graph(directed=False)
    G = simple_graph(G_multi)
    nodes_df, _ = load_nodes_edges()
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Check financial data is present
    n_with_budget = sum(
        1 for _, d in G.nodes(data=True)
        if d.get("node_type") in {"project", "pilot", "prototype"}
        and float(d.get("associated_budget", 0) or 0) > 0
    )
    n_with_invest = sum(
        1 for _, d in G.nodes(data=True)
        if d.get("node_type") == "agent"
        and float(d.get("investment_eur_estimate", 0) or 0) > 0
    )
    print(f"  Initiatives with budget: {n_with_budget}")
    print(f"  Agents with investment estimate: {n_with_invest}")

    if n_with_budget == 0 and n_with_invest == 0:
        print("\n  WARNING: No financial data found on this dataset.")
        print("  Run 00_create_173_synthetic_dataset.py first, then re-run.")
        return

    plot_budget_distribution(nodes_df)

    leverage_df = run_value_leverage(G, nodes_df)
    stranded_df = run_stranded_assets(G, leverage_df)
    diffusion_df = run_financial_diffusion(G_multi, G)
    reallocation_result = run_budget_allocation_simulation(leverage_df)

    summary = {
        "platform_id": PLATFORM_ID,
        "n_initiatives_with_budget": n_with_budget,
        "n_agents_with_investment": n_with_invest,
        "top_leverage": leverage_df.head(5)[["label", "leverage_score", "associated_budget"]].to_dict("records") if not leverage_df.empty else [],
        "stranded_assets_count": len(stranded_df),
        "top_stranded": stranded_df.head(5)[["label", "associated_budget"]].to_dict("records") if not stranded_df.empty else [],
        "budget_reallocation": reallocation_result,
    }
    write_json(ANALYSIS_DIR / "synthetic_financial_summary.json", summary)

    print("\n" + "=" * 60)
    print("FINANCIAL SIMULATION COMPLETE")
    print(f"Plots: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

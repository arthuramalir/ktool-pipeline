from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from structural_change import framework as scf

VD_ORDER = [
    "cultural_identity", "social_justice", "collaboration",
    "innovation_drive", "evidence_based", "community_autonomy",
    "austerity_scarcity",
]

st.set_page_config(
    layout="wide",
    page_title="ALC K-Tool",
    page_icon="K",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
NODE_TYPE_COLORS = {
    "agent": "#1F77B4",
    "project": "#2CA02C",
    "pilot": "#17BECF",
    "prototype": "#BCBD22",
    "information": "#FF7F0E",
    "channel": "#9467BD",
    "perception": "#D62728",
    "challenge": "#8C564B",
    "value": "#E377C2",
    "theme": "#7F7F7F",
    "session": "#AEC7E8",
    "initiative": "#2CA02C",
    "unknown": "#BFBFBF",
}

EDGE_FAMILY_LABELS = {
    "declared_relational": "Declared links [agent↔project, project↔perception, initiative interconnections]",
    "interpretive": "AI-inferred [similarity, contradiction, causality, sequence between quotes]",
    "listening": "Listening [channel → information → value evidence chain]",
    "qualitative_narrative": "Narrative analysis [perception ↔ challenge, pattern ↔ perception links]",
    "quote_semantic": "Quote semantic [legacy — replaced by AI-inferred]",
}

STATUS_COLORS = {
    "Robust": "#2a9d8f",
    "Weak": "#e9c46a",
    "Low coherence": "#f4a261",
    "Single channel": "#e76f51",
    "High internal contradiction": "#e63946",
    "Low purity": "#d62728",
}

PLOTLY_PALETTE = px.colors.qualitative.Set2

BRIDGE_COLORS = {"learning_bridge": "#2ecc71", "coalition_reinforcement": "#3498db",
                 "cleavage_breach": "#e74c3c", "structural_only": "#95a5a6"}
BRIDGE_LABELS = {"learning_bridge": "Learning Bridge", "coalition_reinforcement": "Coalition Reinforcement",
                 "cleavage_breach": "Cleavage Breach", "structural_only": "Structural Only"}
BADGE_COLORS = {"✅ Fundable": "#2ecc71", "⚠️ Cross-domain": "#f39c12", "❓ Topological": "#e74c3c"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def color_for_node_type(nt: str) -> str:
    return NODE_TYPE_COLORS.get(str(nt).strip().lower(), NODE_TYPE_COLORS["unknown"])


def friendly_edge_family(family: str) -> str:
    return EDGE_FAMILY_LABELS.get(str(family).strip(), str(family).replace("_", " ").title())


def fallback_label(label_value, global_id) -> str:
    label_text = "" if pd.isna(label_value) else str(label_value).strip()
    if not label_text or label_text.lower().startswith("unnamed"):
        return str(global_id)
    return label_text


def display_channel_label(row: pd.Series) -> str:
    for column in ["channel_name", "channel_code", "channel_id"]:
        if column in row and pd.notna(row[column]):
            value = str(row[column]).strip()
            if value and value.lower() not in {"unknown channel", "nan", "none", "no channel defined"}:
                return value
    information_id = row.get("information_id", "")
    return f"Channel {information_id}" if information_id else "No channel defined"


@st.cache_data
def load_csv(candidates: list[Path]) -> pd.DataFrame:
    for path in candidates:
        if path.exists() and path.stat().st_size > 2:
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return pd.DataFrame()


@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def confidence_label(failed_checks: int) -> tuple[str, str]:
    if failed_checks <= 1:
        return "High", "success"
    if failed_checks <= 3:
        return "Medium", "warning"
    return "Low", "error"


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_dt(val):
    if pd.isna(val) or not str(val).strip():
        return pd.NaT
    s = str(val).strip().replace("Z", "").split("T")[0]
    try:
        return pd.Timestamp(s)
    except (ValueError, TypeError):
        return pd.NaT


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def network_figure(nodes: pd.DataFrame, edges: pd.DataFrame, max_nodes: int = 120) -> go.Figure:
    if nodes.empty or edges.empty:
        return go.Figure()

    local_edges = edges.copy()
    local_edges["source_global_id"] = local_edges["source_global_id"].astype(str)
    local_edges["target_global_id"] = local_edges["target_global_id"].astype(str)

    degree_counts = pd.concat(
        [local_edges["source_global_id"], local_edges["target_global_id"]], axis=0
    ).value_counts()

    keep_nodes = set(degree_counts.head(max_nodes).index.tolist()) or set(
        nodes["global_id"].astype(str).head(max_nodes)
    )
    nodes_small = nodes[nodes["global_id"].astype(str).isin(keep_nodes)].copy()
    edges_small = local_edges[
        local_edges["source_global_id"].isin(keep_nodes)
        & local_edges["target_global_id"].isin(keep_nodes)
    ].copy()

    graph = nx.Graph()
    for _, row in nodes_small.iterrows():
        graph.add_node(str(row["global_id"]))
    for _, row in edges_small.iterrows():
        graph.add_edge(str(row["source_global_id"]), str(row["target_global_id"]))

    if graph.number_of_nodes() == 0:
        return go.Figure()

    pos = nx.spring_layout(graph, k=0.45, iterations=70, seed=42)

    edge_x, edge_y = [], []
    for src, tgt in graph.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#CFCFCF"),
        hoverinfo="none", mode="lines", name="Connections",
    )

    node_lookup = nodes_small.set_index("global_id").to_dict(orient="index")
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        attrs = node_lookup.get(node, {})
        label = fallback_label(attrs.get("label", node), node)
        ntype = attrs.get("node_type", "unknown")
        degree = graph.degree(node)
        node_text.append(f"{label}<br>Type: {ntype}<br>Links: {degree}")
        node_color.append(color_for_node_type(ntype))
        node_size.append(8 + min(22, degree * 1.2))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers",
        hoverinfo="text", text=node_text,
        marker=dict(size=node_size, color=node_color, line=dict(width=0.8, color="#FFF"), opacity=0.88),
        name="Entities",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        template="plotly_white", showlegend=False,
        margin=dict(l=8, r=8, t=30, b=8),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=560, title="Ecosystem relationship map",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# INITIAL DATA LOAD (nodes/edges needed for date computation)
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

platform_id_from_env = os.environ.get("KTOOL_PLATFORM_ID", "173")
output_subdir_from_env = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    platform_id = st.text_input("Platform ID", value=platform_id_from_env).strip() or "173"
    output_subdir = st.text_input("Output folder", value=output_subdir_from_env).strip() or "test"
    st.caption(f"`data/processed/{platform_id}/{output_subdir}`")

    DATA_DIR = ROOT / "data" / "processed" / platform_id / output_subdir
    ANALYTICS_DIR = DATA_DIR / "analytics"
    ANALYSIS_DIR = DATA_DIR / "analysis"

    nodes = load_csv([ANALYTICS_DIR / "nodes.csv", DATA_DIR / "nodes.csv"])
    edges = load_csv([ANALYTICS_DIR / "edges.csv", DATA_DIR / "edges.csv"])

    if nodes.empty or edges.empty:
        st.error("Nodes/edges not found. Run the graph pipeline first for this platform.")
        st.stop()

    is_synthetic = "synthetic" in platform_id.lower()

    nodes["best_date"] = pd.NaT
    for col in ["date", "created_at", "published_at"]:
        if col in nodes.columns:
            parsed = nodes[col].apply(parse_dt)
            nodes["best_date"] = nodes["best_date"].fillna(parsed)

    observed_min = nodes["best_date"].min()
    observed_max = nodes["best_date"].max()
    if pd.isna(observed_min) or pd.isna(observed_max):
        observed_min = pd.Timestamp("2024-01-01")
        observed_max = pd.Timestamp("2025-12-31")
    default_start = observed_min.date() if hasattr(observed_min, "date") else observed_min
    default_end = observed_max.date() if hasattr(observed_max, "date") else observed_max

    st.markdown("**Time filter**")
    date_range = st.date_input(
        "Date range",
        value=(default_start, default_end),
        min_value=observed_min,
        max_value=observed_max,
    )
    st.caption("Filter to nodes with dates within range")
    st.divider()

    st.markdown("**AI data**")
    ai_mode = st.radio("AI data", ["All data", "Source only"], horizontal=True, help="Toggle AI-generated edges and nodes to see what the pipeline adds vs raw KTool data.")
    st.divider()
    if is_synthetic:
        st.success("Synthetic data — Budget tab active")
    else:
        st.info("Switch to `173_synthetic` for the Budget & Finance tab")

# ─────────────────────────────────────────────────────────────────────────────
# TIME-FILTERED DATA
# ─────────────────────────────────────────────────────────────────────────────
if isinstance(date_range, tuple) and len(date_range) == 2:
    t_start, t_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    t_start, t_end = default_start, default_end

in_window = nodes["best_date"].between(t_start, t_end, inclusive="both")
in_window = in_window | nodes["best_date"].isna()
nodes_win = nodes[in_window].copy()
valid_gids = set(nodes_win["global_id"].astype(str))
edges_win = edges[edges["source_global_id"].isin(valid_gids) & edges["target_global_id"].isin(valid_gids)].copy()

# AI filter
if ai_mode == "Source only":
    if "is_ai_generated" in edges_win.columns:
        edges_win = edges_win[edges_win["is_ai_generated"] != True]
    if "is_ai_generated" in nodes_win.columns:
        nodes_win = nodes_win[nodes_win["is_ai_generated"] != True]

listening = load_csv([DATA_DIR / "listening_pipeline.csv", ANALYTICS_DIR / "listening_pipeline.csv"])

readiness = load_json(ANALYSIS_DIR / "graph_readiness_report.json")
structural = load_json(ANALYSIS_DIR / "structural_summary.json")
robustness = load_json(ANALYSIS_DIR / "robustness_summary.json")
quality_gate = load_json(ANALYSIS_DIR / "relation_quality_gate.json")

robustness_decay = load_csv([ANALYSIS_DIR / "robustness_decay.csv"])
vulnerable_connectors = load_csv([ANALYSIS_DIR / "vulnerable_connectors.csv"])
fragility_nodes = load_csv([ANALYSIS_DIR / "top_fragility_nodes.csv"])
centrality = load_csv([ANALYSIS_DIR / "node_centrality.csv"])
diffusion = load_csv([ANALYSIS_DIR / f"{platform_id}_narrative_diffusion.csv"])
perception_diag = load_csv([ANALYSIS_DIR / "perception_diagnostics.csv"])
perception_narrative = load_csv([ANALYSIS_DIR / "perception_narrative_profiles.csv"])
perception_value_div = load_csv([ANALYSIS_DIR / "perception_value_divergence.csv"])
gnn_summary = load_json(ANALYSIS_DIR / "gnn" / "gnn_summary.json")
gnn_training_report = load_json(ANALYSIS_DIR / "gnn" / "gnn_training_report.json")
gnn_link_report = load_json(ANALYSIS_DIR / "gnn" / "gnn_link_prediction_report.json")
gnn_link_recommendations = load_csv([ANALYSIS_DIR / "gnn" / "gnn_link_recommendations.csv"])
gnn_perception_effects = load_csv([ANALYSIS_DIR / "gnn" / "gnn_perception_effects.csv"])
narrative_profiles = load_csv([ANALYSIS_DIR / "narrative_profiles.csv"])
quote_clusters = load_csv([ANALYSIS_DIR / "quote_clusters.csv"])
manual_profiles = load_json(ROOT / "src" / "analysis" / "manual_narrative_profiles.json")
link_intervention_scores = load_csv([ANALYSIS_DIR / "link_intervention_scores.csv"])
link_intervention_summary = load_json(ANALYSIS_DIR / "link_intervention_summary.json")
link_perception_impact = load_csv([ANALYSIS_DIR / "link_intervention_perception_impact.csv"])

# Narrative impact of GNN-predicted links
impact_edge_candidates = load_csv([ANALYSIS_DIR / "narrative_impact_predictions.csv"])
impact_node_proposals = load_csv([ANALYSIS_DIR / "narrative_impact_node_proposals.csv"])
impact_invented_nodes = load_csv([ANALYSIS_DIR / "narrative_impact_invented_nodes.csv"])
impact_dashboard = load_json(ANALYSIS_DIR / "narrative_impact_dashboard.json")

prototype_candidates = load_csv([ANALYSIS_DIR / "prototype_project_candidates.csv"])
prototype_summary = load_json(ANALYSIS_DIR / "prototype_project_candidates_summary.json")

structural_change = load_json(ANALYSIS_DIR / "structural_change_possibility.json")
change_nodes = load_csv([ANALYSIS_DIR / "change_readiness_nodes.csv"])

structural_hypotheses = load_json(ANALYSIS_DIR / "structural_hypotheses.json")
financial_bridge = load_csv([ANALYSIS_DIR / "financial_perception_bridge.csv"])
narrative_budget = load_csv([ANALYSIS_DIR / "narrative_level_budget_crosstab.csv"])

# Cluster-level FJ narrative simulation (Script 27)
cluster_claim_matrix = load_csv([ANALYSIS_DIR / "cluster_claim_matrix.csv"])
cluster_narrative_baseline = load_csv([ANALYSIS_DIR / "cluster_narrative_baseline.csv"])
cluster_narrative_interventions = load_csv([ANALYSIS_DIR / "cluster_narrative_interventions.csv"])
cluster_narrative_adjacency = load_csv([ANALYSIS_DIR / "cluster_narrative_adjacency.csv"])
claim_nodes = load_csv([ANALYSIS_DIR / "narrative_layers" / "claim_nodes.csv"])

# Financial (synthetic only)
leverage_df = load_csv([ANALYSIS_DIR / "synthetic_value_leverage.csv"])
stranded_df = load_csv([ANALYSIS_DIR / "synthetic_stranded_assets.csv"])
fin_diffusion_df = load_csv([ANALYSIS_DIR / "synthetic_financial_diffusion.csv"])
fin_summary = load_json(ANALYSIS_DIR / "synthetic_financial_summary.json")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("ALC K-Tool: Ecosystem Dashboard")
n_here = len(nodes_win)
e_here = len(edges_win)
n_total = len(nodes)
e_total = len(edges)
caption_counts = f"{n_here:,} / {n_total:,} items · {e_here:,} / {e_total:,} links"
tags = []
if n_here < n_total:
    tags.append("time-filtered")
if ai_mode == "Source only":
    tags.append("AI-filtered")
if tags:
    caption_counts += f" ({', '.join(tags)})"
st.caption(
    f"Platform **{platform_id}** · folder **{output_subdir}** · {caption_counts}"
)

# ─────────────────────────────────────────────────────────────────────────────
# TOPLINE METRICS
# ─────────────────────────────────────────────────────────────────────────────
scope = readiness.get("graph_scope_summary", {})
topology = structural.get("topology_metrics", {})
failed_checks = int(quality_gate.get("failed_checks", 0)) if quality_gate else 0
confidence_text, confidence_level = confidence_label(failed_checks)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Items", f"{int(scope.get('node_count', n_here)):,}", help=f"{n_total} total unfiltered")
c2.metric("Links", f"{int(scope.get('edge_count', e_here)):,}", help=f"{e_total} total unfiltered")
c3.metric("Clusters", int(scope.get("component_count", topology.get("connected_components_count", 0))))
c4.metric("Orphans", int(topology.get("total_isolated_nodes", 0)))
c5.metric("Data confidence", confidence_text)
if not perception_diag.empty:
    robust_count = (perception_diag["status_flag"] == "Robust").sum()
    c6.metric("Solid perceptions", f"{robust_count}/{len(perception_diag)}")

if confidence_level == "error":
    st.error(f"Low confidence ({failed_checks} checks failed)")
elif confidence_level == "warning":
    st.warning(f"Medium confidence ({failed_checks} checks failed)")
else:
    st.success(f"High confidence ({failed_checks} checks failed)")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_labels = [
    "Decision Brief",
    "Overview",
    "Geography & Evolution",
    "Network Layers",
    "Health Check",
    "Listening",
    "AI-Generated Links",
    "Story Clusters",
    "Narrative Space",
    "Perceptions",
    "Claims",
    "GNN Predictions",
    "Narrative Simulation",
    "Structural Change",
]
if is_synthetic:
    tab_labels.append("Budget & Finance")

tabs = st.tabs(tab_labels)
tab_decision = tabs[0]
tab_overview = tabs[1]
tab_geo = tabs[2]
tab_layer = tabs[3]
tab_alerts = tabs[4]
tab_narrative = tabs[5]
tab_ai_semantic = tabs[6]
tab_profiles = tabs[7]
tab_narrative_space = tabs[8]
tab_perception = tabs[9]
tab_claims = tabs[10]
tab_gnn = tabs[11]
tab_narrative_sim = tabs[12]
tab_structural = tabs[13]
tab_financial = tabs[14] if is_synthetic and len(tabs) > 14 else None


auc_val = None
if gnn_training_report:
    test_metrics = gnn_training_report.get("metrics", {}).get("test", {})
    auc_val = float(test_metrics.get("auc", 0)) or float(test_metrics.get("accuracy", 0))

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 — Decision Brief
# ═══════════════════════════════════════════════════════════════════════════════
with tab_decision:
    st.subheader("Dashboard Summary")
    st.caption(
        "Aggregate metrics across all analytical modules. "
        "All figures are based on the currently filtered data."
    )

    # ── Row 1: Core counts ──
    r1a, r1b, r1c, r1d = st.columns(4)
    r1a.metric("Items", f"{n_here:,}", help=f"{n_total} total before time filter")
    r1b.metric("Links", f"{e_here:,}", help=f"{e_total} total before time filter")
    r1c.metric("Node types", f"{nodes_win['node_type'].nunique()}", help="Distinct entity categories")
    r1d.metric("Edge families", f"{edges_win['edge_family'].nunique() if 'edge_family' in edges_win.columns else '?'}", help="Distinct link categories")

    # ── Row 2: Quality ──
    r2a, r2b, r2c, r2d = st.columns(4)
    confidence_text, confidence_level = confidence_label(failed_checks)
    r2a.metric("Data confidence", confidence_text)
    orphan_rate = topology.get("orphan_rate", topology.get("total_isolated_nodes", 0)) / max(1, topology.get("node_count", n_here))
    r2b.metric("Orphan rate", f"{orphan_rate:.0%}", help="Nodes with no connections")
    component_count = int(scope.get("component_count", topology.get("connected_components_count", 0)))
    r2c.metric("Components", f"{component_count:,}", help="Separate connected subgraphs")
    r2d.metric("Edge density", f"{topology.get('edge_density', 0):.4f}", help="Fraction of possible edges that exist")

    # ── Row 3: Structural ──
    st.divider()
    st.markdown("### Ecosystem Readiness *(See Structural Change tab for details)*")
    if structural_change and "change_readiness_scores" in structural_change:
        cr = structural_change["change_readiness_scores"]
        invert = {"blockage_score", "lockin_score"}
        readiness_help = {
            "leverage_score": "Fraction of nodes that are well-connected hubs. Higher = more places where a small change can ripple through the network.",
            "plasticity_score": "How easily the network can rewire without collapsing. Higher = fewer fragile dependencies between parts.",
            "blockage_score": "Fraction of nodes that are single points of failure. Lower = fewer single points of failure.",
            "lockin_score": "How stuck the network is in its current structure. Lower = easier to introduce new connections.",
        }
        for key, label in [("leverage_score", "Leverage ↑ good"),
                           ("plasticity_score", "Plasticity ↑ good"),
                           ("blockage_score", "Blockage ↓ good"),
                           ("lockin_score", "Lock-in ↓ good")]:
            val = float(cr.get(key, 0))
            display_val = 1.0 - val if key in invert else val
            bar_w = max(1, min(100, round(display_val * 100)))
            bar_color = "#2ecc71" if display_val > 0.6 else "#f39c12" if display_val > 0.3 else "#e74c3c"
            lc, mc, rc = st.columns([0.12, 0.76, 0.12])
            with lc:
                st.markdown(f'<span style="font-size: 0.85rem; color: #888;">{label}</span>', unsafe_allow_html=True)
            with mc:
                st.markdown(
                    f'<div style="height: 16px; background: rgba(128,128,128,0.1); border-radius: 8px; overflow: hidden;">'
                    f'<span style="display: block; height: 100%; width: {bar_w}%; '
                    f'background: {bar_color}; border-radius: 8px;"></span></div>',
                    unsafe_allow_html=True,
                )
            with rc:
                st.metric("", f"{val:.2f}", help=readiness_help.get(key, ""), label_visibility="collapsed")
        overall = float(cr.get("overall_readiness", 0))
        st.caption(
            f"Overall change readiness: **{overall:.2f}** — "
            f"{'Network is open to reconfiguration' if overall > 0.5 else 'Structural barriers limit change potential'}. "
            f"Blockages (fragile connectors, small components) need attention."
        )
    else:
        st.info("Run `20_structural_change_possibility.py` to see ecosystem readiness.")

    # ── Row 4: Narrative ──
    st.divider()
    st.markdown("### Narrative Overview")
    nar_cols = st.columns(4)
    n_clusters = len(quote_clusters["cluster_id"].unique()) if not quote_clusters.empty and "cluster_id" in quote_clusters.columns else 0
    n_claims = len(claim_nodes) if not claim_nodes.empty else 0
    n_perceptions = len(perception_diag) if not perception_diag.empty else 0
    robust_perceptions = (perception_diag["status_flag"] == "Robust").sum() if not perception_diag.empty and "status_flag" in perception_diag.columns else 0
    nar_cols[0].metric("Story clusters", n_clusters)
    nar_cols[1].metric("Claims", n_claims)
    nar_cols[2].metric("Perceptions", f"{robust_perceptions}/{n_perceptions} solid" if n_perceptions else 0)
    nar_cols[3].metric("Narrative diffusion bias", f"{diffusion['pr_bias'].max():+.3f}" if not diffusion.empty and 'pr_bias' in diffusion.columns else "—", help="Max over/under-representation across node types")

    # ── Row 5: Narrative simulation convergence ──
    if not cluster_narrative_baseline.empty:
        total_clusters = cluster_narrative_baseline["cluster_id"].nunique()
        total_claims_in_sim = int(cluster_narrative_baseline["claim_count"].sum()) if "claim_count" in cluster_narrative_baseline.columns else 0
        nar_cols2 = st.columns(4)
        nar_cols2[0].metric("Clusters in simulation", total_clusters, help="Clusters with opinion profiles for FJ dynamics")
        nar_cols2[1].metric("Claims feeding model", total_claims_in_sim, help="Total claims aggregated into cluster opinions")
        nar_cols2[2].metric("Intervention scenarios", len(cluster_narrative_interventions) if not cluster_narrative_interventions.empty else 0, help="Target × affected cluster combinations tested")
        nar_cols2[3].metric("Max intervention delta", f"{cluster_narrative_interventions[[c for c in cluster_narrative_interventions.columns if c.startswith('delta_')]].abs().max().max():.4f}" if not cluster_narrative_interventions.empty and any(c.startswith('delta_') for c in cluster_narrative_interventions.columns) else "—", help="Largest opinion shift observed")

    # ── Data quality footnotes ──
    with st.expander("Data quality notes"):
        embed_coverage = float(gnn_summary.get("semantic_embedding_coverage_rate", 0)) if gnn_summary else 0
        st.markdown(
            f"- **GNN validation AUC**: {auc_val:.0%} (likely overfit on {n_here}-node graph)" if auc_val else "- **GNN**: not trained\n"
            f"- **Semantic embedding coverage**: {embed_coverage:.0%} of nodes\n"
            f"- **Date coverage**: {nodes_win['best_date'].notna().sum()}/{n_here} nodes have dates\n"
            f"- **Time range**: {observed_min.date()} to {observed_max.date()}\n"
            f"- **Platform**: {platform_id} / {output_subdir}\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("The network at a glance")
    showing_all = len(nodes_win) == len(nodes)
    if not showing_all:
        st.caption(f"Showing **{len(nodes_win):,}** of **{len(nodes):,}** items (time filter active)")

    left, right = st.columns([0.38, 0.62])
    with left:
        type_counts = nodes_win["node_type"].astype(str).value_counts().reset_index()
        type_counts.columns = ["Node type", "Count"]
        fig_types = px.bar(
            type_counts, x="Count", y="Node type", orientation="h",
            template="plotly_white", title="What's in the network",
            color="Node type", color_discrete_sequence=PLOTLY_PALETTE,
        )
        fig_types.update_layout(height=330, margin=dict(l=10, r=10, t=42, b=10), showlegend=False)
        st.plotly_chart(fig_types, width='stretch')

        if not centrality.empty:
            top_connectors = centrality.copy()
            if "label" in top_connectors.columns and "global_id" in top_connectors.columns:
                top_connectors["label"] = top_connectors.apply(
                    lambda row: fallback_label(row.get("label"), row.get("global_id")), axis=1
                )
            top_connectors = top_connectors.sort_values("betweenness_centrality", ascending=False).head(8)
            st.markdown("**Most connected bridges**")
            st.dataframe(
                top_connectors[["label", "node_type", "betweenness_centrality", "degree"]]
                .rename(columns={"label": "Item", "node_type": "Type",
                                 "betweenness_centrality": "Bridge score", "degree": "Links"}),
                width='stretch', hide_index=True,
            )

    with right:
        max_nodes = st.slider("Items shown on map", 40, 220, 120, 10)
        st.plotly_chart(network_figure(nodes_win, edges_win, max_nodes=max_nodes), width='stretch')

        # Node type legend
        counts = nodes_win["node_type"].fillna("unknown").astype(str).str.lower().value_counts().to_dict()
        legend_items = []
        for nt in sorted(counts, key=lambda t: -counts[t]):
            color = color_for_node_type(nt)
            legend_items.append(
                f"<span style='display:inline-block;width:12px;height:12px;"
                f"background:{color};border-radius:2px;margin-right:6px;vertical-align:middle;'></span>"
                f"<strong>{nt.title()}</strong> ({counts[nt]})"
            )
        if legend_items:
            st.markdown("<br>".join(legend_items), unsafe_allow_html=True)

    if not showing_all:
        st.divider()
        st.markdown("**Evolution snapshot**")
        evo1, evo2 = st.columns(2)
        with evo1:
            before = len(nodes)
            after = len(nodes_win)
            lost = before - after
            st.metric("Excluded by time filter", f"{lost:,}", delta=f"{-lost:,}")
        with evo2:
            pct = f"{after / before * 100:.1f}%" if before > 0 else "N/A"
            st.metric("Remaining", pct)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Health Check
# ═══════════════════════════════════════════════════════════════════════════════
with tab_alerts:
    st.subheader("Network health check")

    targeted_drop = to_float(robustness.get("systemic_drop_delta_targeted", 0.0))
    random_drop = to_float(robustness.get("systemic_drop_delta_random", 0.0))
    verdict = robustness.get("verdict", "No robustness verdict found for this run.")

    left, right = st.columns([0.5, 0.5])
    with left:
        st.markdown("**Stress test results**")
        m1, m2 = st.columns(2)
        m1.metric("Hit to core (targeted)", f"{targeted_drop:.1%}",
                  help="How much the biggest cluster shrinks when the most central nodes are removed.")
        m2.metric("Hit to core (random)", f"{random_drop:.1%}",
                  help="Same, but removing random nodes instead.")
        if targeted_drop > random_drop + 0.2:
            st.error(f"{verdict}")
        else:
            st.info(verdict)

        if not robustness_decay.empty:
            fig_decay = px.line(
                robustness_decay,
                x="nodes_removed_count",
                y=[c for c in ["targeted_attack_core_residual_share", "random_failure_core_residual_share"] if c in robustness_decay.columns],
                template="plotly_white",
                title="How the core holds up",
                labels={"value": "Core remaining", "nodes_removed_count": "Items removed"},
            )
            fig_decay.update_layout(height=320, legend_title="Scenario")
            st.plotly_chart(fig_decay, width='stretch')

    with right:
        st.markdown("**Priority: weak spots to watch**")
        connector_view = vulnerable_connectors.copy() if not vulnerable_connectors.empty else fragility_nodes.copy()
        if connector_view.empty:
            st.info("No vulnerability table available.")
        else:
            if "label" in connector_view.columns and "global_id" in connector_view.columns:
                connector_view["label"] = connector_view.apply(
                    lambda row: fallback_label(row.get("label"), row.get("global_id")), axis=1
                )
            display_cols = [c for c in ["label", "node_type", "score", "confidence_flag", "interpretation"] if c in connector_view.columns]
            st.dataframe(
                connector_view[display_cols].head(12).rename(columns={
                    "label": "Entity", "node_type": "Type", "score": "Risk score",
                    "confidence_flag": "Confidence", "interpretation": "Why it matters",
                }),
                width='stretch', hide_index=True,
            )

    st.markdown("**Recommended actions**")
    action_lines = []
    if targeted_drop > random_drop + 0.2:
        action_lines.append("Add backup bridges for fragile items: aim for at least 3 links each.")
    orphan_project_rate = readiness.get("orphan_node_distributions", {}).get("project", {}).get("orphan_rate", 0.0)
    if to_float(orphan_project_rate) > 0.5:
        action_lines.append("Reconnect orphan projects: many sit outside the main network.")
    if failed_checks > 0:
        action_lines.append("Fill in missing link types before big decisions.")
    if not action_lines:
        action_lines.append("All looks stable. Re-run after each data refresh.")
    for idx, line in enumerate(action_lines, start=1):
        st.write(f"{idx}. {line}")

    with st.expander("How these numbers work"):
        st.markdown(
            "- **Targeted drop** = how much the biggest cluster shrinks after removing the 10% most central nodes.\n"
            "- **Random drop** = same, removing 10% at random.\n"
            "- **If targeted − random > 0.2** → too many eggs in one basket.\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Listening
# ═══════════════════════════════════════════════════════════════════════════════
with tab_narrative:
    st.subheader("What people are saying")

    c1, c2 = st.columns([0.48, 0.52])
    with c1:
        if not listening.empty:
            listening = listening.copy()
            listening["channel_display"] = listening.apply(display_channel_label, axis=1)
            channel_counts = listening["channel_display"].fillna("Unknown").astype(str).value_counts().reset_index()
            channel_counts.columns = ["Channel", "Narratives"]
            fig_ch = px.bar(
                channel_counts.head(12), x="Narratives", y="Channel", orientation="h",
                template="plotly_white", title="Stories by channel",
            )
            fig_ch.update_layout(height=340, margin=dict(l=10, r=10, t=42, b=10))
            st.plotly_chart(fig_ch, width='stretch')
        else:
            st.info("Listening data not available.")

        if not diffusion.empty and "pagerank" in diffusion.columns:
            diff_view = diffusion.copy()
            if "label" in diff_view.columns and "global_id" in diff_view.columns:
                diff_view["label"] = diff_view.apply(
                    lambda row: fallback_label(row.get("label"), row.get("global_id")), axis=1
                )
            top_pr = diff_view.sort_values("pagerank", ascending=False).head(12)
            fig_pr = px.bar(
                top_pr, x="pagerank", y="label", orientation="h",
                template="plotly_white", title="Top PageRank — most central voices",
                color_discrete_sequence=["#2ecc71"],
            )
            fig_pr.update_layout(height=280, margin=dict(l=10, r=10, t=42, b=10), yaxis_title="")
            st.plotly_chart(fig_pr, width='stretch')

            if "diffusion_bias" in diff_view.columns:
                top_diff = diff_view.sort_values("diffusion_bias", ascending=False).head(12)
                fig_diff = px.bar(
                    top_diff, x="diffusion_bias", y="label", orientation="h",
                    template="plotly_white", title="Diffusion bias — voices over/under-represented",
                    color="diffusion_bias", color_continuous_scale="RdYlBu", range_color=[-1, 1],
                )
                fig_diff.update_layout(height=280, margin=dict(l=10, r=10, t=42, b=10), yaxis_title="", coloraxis_showscale=False)
                st.plotly_chart(fig_diff, width='stretch')
        elif not diffusion.empty:
            st.info("Diffusion data loaded but PageRank column missing.")
        else:
            st.info("PageRank data not available. Run `26_compute_narrative_diffusion.py` first.")

        with c2:
            st.markdown("**Browse quotes**")
            if listening.empty:
                st.info("No quotes available.")
            else:
                listening = listening.copy()
                listening["channel_display"] = listening.apply(display_channel_label, axis=1)
                channels = sorted(listening["channel_display"].fillna("Unknown").astype(str).unique().tolist())
                selected_channel = st.selectbox("Filter by channel", options=["All"] + channels)
                view = listening.copy()
                if selected_channel != "All":
                    view = view[view["channel_display"].astype(str) == selected_channel]
                show_cols = [c for c in ["information_id", "channel_display", "pattern_names", "value_names", "information_text"] if c in view.columns]
                st.dataframe(
                    view[show_cols].head(40).rename(columns={
                        "information_id": "ID", "channel_display": "Channel",
                        "pattern_names": "Patterns", "value_names": "Values", "information_text": "Quote",
                    }),
                    width='stretch', hide_index=True,
                )

    with st.expander("How influence is measured"):
        st.markdown(
            "- **PageRank** scores how central each item is in the conversation network (random walk through the graph).\n"
            "- **Diffusion bias** = personalized PageRank comparing two largest agent types. Positive = voice travels further in the [type A] network. Negative = voice travels further in the [type B] network.\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Story Clusters
# ═══════════════════════════════════════════════════════════════════════════════
with tab_profiles:
    st.subheader("Story clusters from listening data")

    st.markdown(
        "Each **story cluster** groups related quotes together. The system finds these automatically "
        "by measuring how similar quotes are to each other. Below that, you'll see **three-layer profiles** "
        "(surface / hidden / big-picture) for the key stories."
    )

    # ── Narrative × perception topic landscape visualization
    st.markdown("### Topics across story clusters")
    st.caption("Each bubble is a topic in a story cluster. Bigger = more quotes.")

    if not quote_clusters.empty:
        topic_data = []
        for _, row in quote_clusters.iterrows():
            topics = str(row.get("topics_thematic_areas", "")).split(";")
            for t in topics:
                t = t.strip()
                if t:
                    topic_data.append({"cluster_id": row["cluster_id"], "topic": t})
        topic_df = pd.DataFrame(topic_data)
        if not topic_df.empty:
            topic_matrix = topic_df.groupby(["cluster_id", "topic"]).size().reset_index(name="count")
            cluster_labels = dict(zip(narrative_profiles["cluster_id"], narrative_profiles["cluster_label"])) if not narrative_profiles.empty else {}
            topic_matrix["cluster"] = topic_matrix["cluster_id"].map(cluster_labels).fillna(topic_matrix["cluster_id"])

            # Bubble chart: x=cluster, y=topic, size=count
            fig_np = px.scatter(
                topic_matrix, x="cluster", y="topic", size="count",
                    color="topic", color_discrete_sequence=px.colors.qualitative.Set2,
                template="plotly_white",
                title="What each story cluster talks about",
                labels={"cluster": "Story cluster", "topic": "Topic", "count": "Quote count"},
            )
            fig_np.update_layout(height=400, showlegend=False, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_np, width='stretch')

            # Topic-perception overlay
            st.caption("Topics that appear in multiple clusters.")
            topic_pivot = topic_df.groupby("topic")["cluster_id"].nunique().reset_index(name="n_profiles")
            topic_pivot = topic_pivot.sort_values("n_profiles", ascending=False).head(15)
            fig_topic = px.bar(
                topic_pivot, x="n_profiles", y="topic", orientation="h",
                template="plotly_white", color="n_profiles", color_continuous_scale="Blues",
                title="Topics that cross story clusters",
                labels={"n_profiles": "Clusters", "topic": "Topic"},
            )
            fig_topic.update_layout(height=320, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_topic, width='stretch')
    else:
        st.info("Run the quote clustering pipeline first.")

    st.divider()

    if isinstance(manual_profiles, list) and manual_profiles:
        for prof in manual_profiles:
            with st.expander(f"**{prof.get('name', 'Unnamed')}** — channels: {', '.join(prof.get('source_channels', []))}"):
                c1, c2 = st.columns([0.5, 0.5])
                with c1:
                    st.markdown(f"**Key idea** — {prof.get('key_idea', '')}")
                    st.markdown(f"**Surface story:** {prof.get('surface_narrative', '')}")
                    st.markdown(f"**Hidden layer:** {prof.get('implicit_narrative', '')}")
                    st.markdown(f"**Big picture:** {prof.get('metanarrative', '')}")
                with c2:
                    st.markdown("**What people said:**")
                    for q in prof.get("representative_quotes", []):
                        st.markdown(f"- _{q}_")
                    st.markdown(f"**Tone:** {prof.get('emotional_tone', '')}")
                    st.markdown(f"**Values:** {', '.join(prof.get('associated_values', []))}")
                    if prof.get("contradiction_with"):
                        st.markdown(f"**Opposes:** {', '.join(prof['contradiction_with'])}")
    else:
        st.info("No manual story profiles found. Add profiles to `src/analysis/manual_narrative_profiles.json`.")

    st.divider()

    # ── Surface / Implicit / Meta-narrative illustration
    with st.expander("📊 Three-layer story analysis — quick guide"):
        st.markdown(
            "Each story can be read at three levels."
        )
        st.markdown("---")

        col_s, col_i, col_m = st.columns(3)
        with col_s:
            st.markdown("**🔹 Surface**")
            st.markdown("*What people say*")
            st.success(
                "The direct statement. No reading between the lines."
            )
            st.markdown("How to spot it:")
            st.markdown("- Paraphrase the quote in one sentence\n- Stick to the words used")
            st.markdown("---")
            st.info("_Example:_\n'Short-term funding makes it hard to plan long-term projects.'")

        with col_i:
            st.markdown("**🔸 Hidden**")
            st.markdown("*What's assumed*")
            st.warning(
                "The belief or assumption underneath. Ask: 'what must be true for this to make sense?'"
            )
            st.markdown("How to spot it:")
            st.markdown("- Ask what's taken for granted\n- Look for unspoken causes")
            st.markdown("---")
            st.info("_Example:_\n'Short-term funding = funders don't trust communities with multi-year budgets.'")

        with col_m:
            st.markdown("**🔶 Big picture**")
            st.markdown("*The deeper pattern*")
            st.error(
                "The wider context or power dynamic. Ask: 'what system does this reinforce or push against?'"
            )
            st.markdown("How to spot it:")
            st.markdown("- Link to the broader context\n- Ask whose interests it serves")
            st.markdown("---")
            st.info("_Example:_\n'Short-term funding looks like accountability but actually makes long-term planning impossible — a way to cut core support while appearing responsible.'")

        st.divider()
        st.markdown(
            "**How to write a three-layer profile:**\n\n"
            "1. Start with **surface** — what the quote actually says.\n"
            "2. Ask **'what's taken for granted?'** to find the **hidden** layer.\n"
            "3. Ask **'what bigger pattern does this fit?'** to reach the **big picture**.\n"
        )

    st.divider()
    st.markdown("### Auto-detected story clusters")

    if not narrative_profiles.empty:
        st.caption(f"{len(narrative_profiles)} clusters from {narrative_profiles['quote_count'].sum()} quotes.")
        for _, row in narrative_profiles.iterrows():
            with st.expander(f"**{row.get('cluster_label', 'Cluster')}** ({int(row.get('quote_count', 0))} quotes)"):
                st.markdown(f"**Top channels:** {row.get('top_channels', '—')}")
                st.markdown(f"**Representative quote:** _{row.get('representative_quote', '—')}_")
                st.markdown(f"**Avg selection score:** {row.get('average_selection_score', '—')}")
    else:
        st.info("No auto-generated profiles found. Run `17_cluster_quotes_into_profiles.py`.")

    if not quote_clusters.empty:
        with st.expander("All quote-cluster assignments"):
            view = quote_clusters[["cluster_id", "information_id", "quote", "channel_code"]].copy()
            st.dataframe(view, width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Narrative Space (unified view)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_narrative_space:
    st.subheader("Narrative space")
    st.caption("Unified view of story clusters, claims, and perceptions — every quote projected by semantic similarity.")

    # Prepare data
    ns_clusters = quote_clusters.copy() if not quote_clusters.empty else pd.DataFrame()
    ns_matrix = cluster_claim_matrix.copy() if not cluster_claim_matrix.empty else pd.DataFrame()

    if ns_clusters.empty:
        st.info("No quote clusters available. Run `17_cluster_quotes_into_profiles.py` first.")
    else:
        # TF-IDF embedding of all quotes for the 2D projection
        texts = ns_clusters["quote"].fillna("").astype(str).tolist()
        vec = TfidfVectorizer(stop_words="english", max_features=500)
        tfidf_mat = vec.fit_transform(texts)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(tfidf_mat.toarray())

        plot_df = ns_clusters[["cluster_id", "information_id", "quote", "channel_code"]].copy()
        plot_df["x"] = coords[:, 0]
        plot_df["y"] = coords[:, 1]

        # Merge claim dimensions if available
        has_claim_col = "claim_id" in ns_matrix.columns
        claim_info = ns_matrix[ns_matrix["claim_id"].notna()][["information_id", "value_dimension", "narrative_level", "contradiction_flag"]].drop_duplicates("information_id") if has_claim_col and not ns_matrix.empty else pd.DataFrame()
        if not claim_info.empty:
            plot_df = plot_df.merge(claim_info, on="information_id", how="left")

        ns_left, ns_right = st.columns([0.62, 0.38])

        with ns_left:
            st.markdown("**Semantic map of all quotes** — coloured by cluster, shaped by type")

            # Aggregate cluster stats for legend
            cluster_stats = ns_clusters.groupby("cluster_id").agg(
                count=("information_id", "nunique")
            ).reset_index()

            # Build cluster palette
            all_clusters = sorted(plot_df["cluster_id"].unique())
            cluster_palette = {c: PLOTLY_PALETTE[i % len(PLOTLY_PALETTE)] for i, c in enumerate(all_clusters)}

            fig_ns = go.Figure()

            for cid in all_clusters:
                sub = plot_df[plot_df["cluster_id"] == cid]
                n = cluster_stats[cluster_stats["cluster_id"] == cid]["count"].values[0]
                fig_ns.add_trace(go.Scattergl(
                    x=sub["x"].tolist(), y=sub["y"].tolist(),
                    mode="markers",
                    marker=dict(size=8, color=cluster_palette[cid], opacity=0.8),
                    text=sub["quote"].tolist(),
                    customdata=sub[["cluster_id", "information_id"]].fillna("").values.tolist(),
                    hovertemplate="<b>Cluster %{customdata[0]}</b><br>%{text[:200]}<extra></extra>",
                    name=f"{cid} ({n} quotes)",
                    showlegend=True,
                ))

            fig_ns.update_layout(
                template="plotly_white",
                height=500,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(showticklabels=False, title=""),
                yaxis=dict(showticklabels=False, title=""),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_ns, use_container_width=True)

            st.markdown(f"**{len(plot_df)} quotes** across **{len(all_clusters)} clusters** · "
                        f"PCA projection of TF-IDF vectors (explained variance: {pca.explained_variance_ratio_.sum():.0%})")

        with ns_right:
            sel_cluster = st.selectbox("Inspect a cluster", options=["All"] + all_clusters)

            if sel_cluster == "All":
                view_df = ns_clusters
            else:
                view_df = ns_clusters[ns_clusters["cluster_id"] == sel_cluster]

            st.markdown(f"**{len(view_df)} quotes in selected view**")

            # Show claim summary for this cluster
            if not ns_matrix.empty and sel_cluster != "All" and "claim_id" in ns_matrix.columns:
                cluster_claims = ns_matrix[ns_matrix["cluster_id"] == sel_cluster]
                claim_rows = cluster_claims[cluster_claims["claim_id"].notna()]
                if not claim_rows.empty:
                    st.markdown("**Claims in this cluster**")
                    dim_counts = claim_rows["value_dimension"].value_counts()
                    st.caption("Value dimensions: " + " · ".join(f"{d} ({c})" for d, c in dim_counts.items() if d))
                    contradictions = claim_rows[claim_rows["contradiction_flag"] == True]
                    if not contradictions.empty:
                        st.warning(f"⚠️ Contradictory claims detected — opposing value dimensions within this cluster")
                    for _, r in claim_rows.head(5).iterrows():
                        lbl = r.get("claim_id", "")
                        dim = r.get("value_dimension", "")
                        lvl = r.get("narrative_level", "")
                        st.caption(f"• {lbl} — *{dim}* ({lvl})")
                else:
                    st.caption("No claims linked to this cluster.")

            # Show quotes
            with st.expander("View quotes", expanded=True):
                for _, r in view_df.iterrows():
                    st.markdown(f"> {r['quote']}")
                    st.caption(f"*{r.get('channel_code', '')}* — {r['cluster_id']}")
                    st.divider()

            # Cluster-to-cluster relationship mini-map
            if not ns_matrix.empty and sel_cluster == "All" and "claim_id" in ns_matrix.columns and "value_dimension" in ns_matrix.columns:
                st.markdown("**How clusters relate**")
                cluster_rels = ns_matrix[ns_matrix["claim_id"].notna()].groupby("cluster_id")["value_dimension"].apply(lambda x: x.value_counts().to_dict()).reset_index()
                st.caption("Clusters sharing value dimensions — clusters with overlapping value profiles are semantically close.")
                rel_data = []
                dim_to_clusters = {}
                for _, r in cluster_rels.iterrows():
                    if isinstance(r["value_dimension"], dict):
                        for dim in r["value_dimension"]:
                            if dim and str(dim).strip():
                                dim_to_clusters.setdefault(str(dim).strip(), []).append(r["cluster_id"])
                shared = [(d, cs) for d, cs in dim_to_clusters.items() if len(cs) > 1]
                if shared:
                    for dim, cs in shared:
                        st.caption(f"**{dim}** appears in: {', '.join(cs)}")
                else:
                    st.caption("No shared value dimensions between clusters.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9 — Perceptions
# ═══════════════════════════════════════════════════════════════════════════════
with tab_perception:
    st.subheader("Perception health check")
    st.caption(
        "How solid each perception is, based on how much the quotes agree."
    )

    if perception_diag.empty:
        st.warning(
            "No perception diagnostics found. Run:\n\n"
            "```\npython src/analysis/09_perception_diagnostics.py\n```"
        )
    else:
        # ── Summary metric row
        n_perceptions = len(perception_diag)
        n_robust = (perception_diag["status_flag"] == "Robust").sum() if "status_flag" in perception_diag.columns else 0
        avg_coherence = perception_diag["internal_coherence"].dropna().mean() if "internal_coherence" in perception_diag.columns else None
        avg_purity = perception_diag["purity_score"].dropna().mean() if "purity_score" in perception_diag.columns else None

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Perceptions", n_perceptions)
        m2.metric("Solid", f"{n_robust}/{n_perceptions}")
        m3.metric("Avg agreement", f"{avg_coherence:.2f}" if avg_coherence is not None else "n/a",
                  help="How similar the quotes are inside each perception. Above 0.6 = strong agreement.")
        m4.metric("Avg focus", f"{avg_purity:.2f}" if avg_purity is not None else "n/a",
                  help="Of each quote's top-5 neighbours, how many share the same perception label.")

        st.divider()

        # ── Colour-coded status table
        st.markdown("**Perception health table** (by quote count)")

        def _status_colour(flag: str) -> str:
            if flag == "Robust":
                return "background-color: #d4edda; color: #155724;"
            if "Weak" in flag:
                return "background-color: #fff3cd; color: #856404;"
            if "Low coherence" in flag:
                return "background-color: #fde8d8; color: #823b00;"
            if "contradiction" in flag.lower():
                return "background-color: #f8d7da; color: #721c24;"
            if "Single channel" in flag:
                return "background-color: #e2d9f3; color: #4a235a;"
            return "background-color: #e2e3e5; color: #383d41;"

        diag_view = perception_diag.copy()
        display_rename = {
            "perception_label": "Perception",
            "quote_count": "Quotes",
            "n_channels": "Channels",
            "internal_coherence": "Agreement ↑",
            "purity_score": "Focus ↑",
            "source_entropy": "Source spread",
            "contradiction_density": "Disagreement %",
            "status_flag": "Status",
        }
        diag_view = diag_view[[c for c in display_rename if c in diag_view.columns]].rename(columns=display_rename)

        # Format floats
        for col in ["Agreement ↑", "Focus ↑", "Source spread", "Disagreement %"]:
            if col in diag_view.columns:
                diag_view[col] = diag_view[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

        # Style status column
        def _style_row(row):
            flag = row.get("Status", "")
            style = _status_colour(str(flag))
            return [style if col == "Status" else "" for col in row.index]

        styled = diag_view.style.apply(_style_row, axis=1)
        st.dataframe(styled, width='stretch', hide_index=True)

        st.divider()

        # ── Scatter: Coherence vs Purity
        if "internal_coherence" in perception_diag.columns and "purity_score" in perception_diag.columns:
            scatter_df = perception_diag.dropna(subset=["internal_coherence", "purity_score"]).copy()
            if "perception_label" in scatter_df.columns:
                scatter_df["label"] = scatter_df["perception_label"].astype(str).str[:40]
            scatter_df["status_flag"] = scatter_df["status_flag"].fillna("Unknown")

            fig_scatter = px.scatter(
                scatter_df,
                x="internal_coherence",
                y="purity_score",
                text="label" if "label" in scatter_df.columns else None,
                color="status_flag",
                size="quote_count" if "quote_count" in scatter_df.columns else None,
                template="plotly_white",
                title="Agreement vs Focus",
                labels={"internal_coherence": "Agreement (how similar quotes are)",
                        "purity_score": "Focus (are quotes close to other perceptions?)"},
                color_discrete_sequence=PLOTLY_PALETTE,
            )
            fig_scatter.add_vline(x=0.4, line_dash="dot", line_color="orange",
                                  annotation_text="Agreement threshold", annotation_position="top left")
            fig_scatter.add_hline(y=0.3, line_dash="dot", line_color="orange",
                                  annotation_text="Focus threshold", annotation_position="bottom right")
            fig_scatter.update_traces(textposition="top center")
            fig_scatter.update_layout(height=480, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_scatter, width='stretch')

        # ── Source entropy bar chart
        if "source_entropy" in perception_diag.columns and "perception_label" in perception_diag.columns:
            ent_df = perception_diag[["perception_label", "source_entropy", "n_channels"]].dropna().copy()
            ent_df["perception_label"] = ent_df["perception_label"].astype(str).str[:50]
            fig_entropy = px.bar(
                ent_df.sort_values("source_entropy"),
                x="source_entropy", y="perception_label", orientation="h",
                template="plotly_white",
                title="Source diversity per perception",
                labels={"source_entropy": "Channel diversity (higher = more sources)",
                        "perception_label": "Perception"},
                color="n_channels", color_continuous_scale="Blues",
            )
            fig_entropy.add_vline(x=0.5, line_dash="dot", line_color="red",
                                  annotation_text="Single-source risk threshold")
            fig_entropy.update_layout(height=360, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_entropy, width='stretch')

        st.divider()

        # ── PHASE 2: Narrative alignment ──
        st.markdown("**Narrative alignment — how perceptions map to the claim space**")
        st.caption(
            "Claims are extracted from each perception's quote text. These metrics show "
            "whether the claims and quotes agree on what the perception is about."
        )

        if not perception_narrative.empty:
            nalign_rename = {
                "perception_label": "Perception",
                "n_quotes": "Quotes",
                "n_claims": "Claims",
                "silhouette_score": "Silhouette ↑",
                "nearest_other_perception": "Nearest neighbour",
                "nearest_other_similarity": "Neighbour similarity",
                "vd_n_dims": "Value dims",
                "vd_entropy": "VD diversity",
                "quote_claim_cosim": "Quote↔Claim alignment ↑",
                "cross_leakage": "Cross-leakage ↓",
            }
            nalign_view = perception_narrative[[c for c in nalign_rename if c in perception_narrative.columns]].copy().rename(columns=nalign_rename)

            for col in ["Silhouette ↑", "Neighbour similarity", "Quote↔Claim alignment ↑", "Cross-leakage ↓", "VD diversity"]:
                if col in nalign_view.columns:
                    nalign_view[col] = nalign_view[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) and x != "none" else "—")

            # Colour silhouette: green > 0, amber ~0, red < 0
            def _sil_colour(val):
                try:
                    v = float(val)
                    if v > 0.1:
                        return "background-color: #d4edda; color: #155724;"
                    if v > -0.1:
                        return "background-color: #fff3cd; color: #856404;"
                    return "background-color: #f8d7da; color: #721c24;"
                except (ValueError, TypeError):
                    return ""

            def _style_nalign(row):
                sil = str(row.get("Silhouette ↑", ""))
                return [_sil_colour(sil) if col == "Silhouette ↑" else "" for col in row.index]

            styled_nalign = nalign_view.style.apply(_style_nalign, axis=1)
            st.dataframe(styled_nalign, width='stretch', hide_index=True)

            # Narrative bar chart: silhouette per perception
            sil_plot = perception_narrative[perception_narrative["silhouette_score"].notna()].copy()
            if not sil_plot.empty:
                sil_plot["perception_label"] = sil_plot["perception_label"].astype(str).str[:50]
                fig_sil = px.bar(
                    sil_plot.sort_values("silhouette_score"),
                    x="silhouette_score", y="perception_label", orientation="h",
                    template="plotly_white",
                    title="Narrative distinctness (silhouette)",
                    labels={"silhouette_score": "Silhouette (higher = more distinct)",
                            "perception_label": ""},
                    color="silhouette_score",
                    color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
                    range_color=(-0.3, 0.3),
                )
                fig_sil.add_vline(x=0, line_dash="dot", line_color="grey",
                                  annotation_text="Not distinct", annotation_position="top right")
                fig_sil.update_layout(height=300, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_sil, width='stretch')

            # Quote-claim alignment bar chart
            qc_plot = perception_narrative[perception_narrative["quote_claim_cosim"].notna()].copy()
            if not qc_plot.empty:
                qc_plot["perception_label"] = qc_plot["perception_label"].astype(str).str[:50]
                fig_qc = px.bar(
                    qc_plot.sort_values("quote_claim_cosim"),
                    x="quote_claim_cosim", y="perception_label", orientation="h",
                    template="plotly_white",
                    title="Do quotes and claims agree? (TF-IDF cosine similarity)",
                    labels={"quote_claim_cosim": "Quote↔Claim alignment",
                            "perception_label": ""},
                    color="quote_claim_cosim",
                    color_continuous_scale=["#f39c12", "#2ecc71"],
                )
                fig_qc.add_vline(x=0.3, line_dash="dot", line_color="orange",
                                 annotation_text="Low alignment", annotation_position="top right")
                fig_qc.update_layout(height=300, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_qc, width='stretch')

            # Narratively redundant pairs
            if not perception_value_div.empty:
                red = perception_value_div[perception_value_div["narratively_redundant"] == True]
                if len(red) > 0:
                    real_red = red[red["perception_a_label"] != "__unassigned__"]
                    if len(real_red) > 0:
                        st.warning(
                            f"**{len(real_red)} narratively redundant pair(s)** — "
                            "these perceptions produce nearly identical claims. "
                            "They may be expressing the same narrative position."
                        )
                        st.dataframe(
                            real_red[["perception_a_label", "perception_b_label", "claim_cosine_similarity"]].rename(columns={
                                "perception_a_label": "Perception A",
                                "perception_b_label": "Perception B",
                                "claim_cosine_similarity": "Claim similarity",
                            }),
                            width='stretch', hide_index=True,
                        )
                    else:
                        st.success("No narratively redundant perceptions — all produce distinct claim profiles.")
                else:
                    st.success("No narratively redundant perceptions — all produce distinct claim profiles.")

        st.info(
            "**How to read these:**\n\n"
            "- **Silhouette > 0** → Perception's quotes are more similar to each other than to other perceptions' quotes. Distinct.\n"
            "- **Silhouette < 0** → Perception's quotes are closer to OTHER perceptions. May be mis-labeled.\n"
            "- **Quote↔Claim alignment > 0.3** → The claims extracted from this perception match its quote content.\n"
            "- **Cross-leakage = 100%** → All top-5 semantic neighbours belong to other perceptions. Low distinctness.\n"
            "- **Narratively redundant** → Two perceptions produce the same claims. Consider merging."
        )

        st.divider()

        st.info(
            "**How to read diagnostics:**\n\n"
            "- **Agreement < 0.4** → Quotes don't agree much. Maybe split into two perceptions.\n"
            "- **Focus < 0.3** → Quotes are closer to OTHER perceptions than their own. May be mis-labeled.\n"
            "- **Source diversity < 0.5** → Only one channel feeds this perception. Could be one person's view, not a shared one.\n"
            "- **Disagreement > 30%** → Strongly contested perception. Important to show, but not a single unified view."
        )

        with st.expander("How each number is calculated"):
            st.markdown(
                "- **Agreement** = how similar the quotes are to each other inside a perception. >0.6 = high agreement.\n"
                "- **Focus** = of each quote's 5 closest neighbours, how many share the same perception. <0.3 = fuzzy boundary.\n"
                "- **Source diversity** = how many different channels feed this perception. <0.5 = one channel dominates.\n"
                "- **Disagreement** = fraction of internal links marked as contradiction.\n"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 11 — GNN Predictions
# ═══════════════════════════════════════════════════════════════════════════════
with tab_gnn:
    st.subheader("What-if: adding new links")
    st.caption(
        "The GNN detects missing connections between agents, projects, and channels based on network structure. "
        "Each suggested link is scored by: **(a)** GNN probability (structural plausibility), "
        "**(b)** domain compatibility (embedding similarity), and "
        "**(c)** narrative impact (new claim pathways unlocked)."
    )

    # ── 1. Suggested Links Table ──
    if not impact_edge_candidates.empty:
        show_cols = ["source_label", "target_label", "link_probability", "gnn_rationale", "bridge_type", "feasibility_badge", "new_narrative_pathways", "new_vds_unlocked"]
        show_cols = [c for c in show_cols if c in impact_edge_candidates.columns]
        link_table = impact_edge_candidates[show_cols].copy()
        link_table.columns = [c.replace("_", " ").title() for c in show_cols]
        st.markdown("**1. Suggested links**")
        st.dataframe(link_table, width='stretch', hide_index=True)
        st.caption(
            "**How to read**: Probability = how strongly the GNN believes this link should exist (0-1). "
            "✅ Fundable = structurally sound + domain-compatible. ❓ Topological = structurally plausible but domain-distant. "
            "**Bridge types**: Learning Bridge = unlocks new value dimensions across separate narrative spaces. "
            "Coalition Reinforcement = strengthens existing shared values. Structural Only = no narrative effect."
        )

    # ── 2. Narrative Impact ──
    if not impact_edge_candidates.empty:
        st.divider()
        st.markdown("**2. Narrative impact — what each link changes**")
        st.caption("For each suggested link: which claims/narratives it touches on both sides, and what new connections it would create.")

        for idx, (_, row) in enumerate(impact_edge_candidates.iterrows()):
            src = row.get("source_label", row.get("source_global_id", "?"))
            tgt = row.get("target_label", row.get("target_global_id", "?"))
            st_src_t = row.get("source_node_type", "")
            st_tgt_t = row.get("target_node_type", "")
            bridge = row.get("bridge_type", "structural_only")
            color = BRIDGE_COLORS.get(bridge, "#95a5a6")
            bl = BRIDGE_LABELS.get(bridge, bridge)
            prob = float(row.get("link_probability", 0))
            badge = row.get("feasibility_badge", "")
            badge_color = {"✅ Fundable": "#2ecc71", "⚠️ Cross-domain": "#f39c12", "❓ Topological": "#e74c3c"}.get(badge, "#95a5a6")
            rationale = row.get("gnn_rationale", "")
            note = row.get("bridge_note", "")
            action = row.get("action_template", "")
            pathways = int(row.get("new_narrative_pathways", 0))
            new_vds = row.get("new_vds_unlocked", "none")
            merges = row.get("merges_components", False)
            src_cnt = int(row.get("source_perception_claim_count", 0))
            tgt_cnt = int(row.get("target_perception_claim_count", 0))
            src_vds = row.get("source_vds", "none")
            tgt_vds = row.get("target_vds", "none")

            st.markdown(
                f'<div style="border-left: 4px solid {color}; padding: 0.5rem 1rem; margin: 0.3rem 0; '
                f'background: rgba(128,128,128,0.03); border-radius: 4px;">'
                f'<div style="display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;">'
                f'<strong>#{idx+1}</strong> '
                f'<span style="background: {color}; color: white; padding: 0.1rem 0.5rem; border-radius: 3px; font-size: 0.8rem;">{bl}</span>'
                f'<span style="font-size: 0.85rem; color: #aaa;">p={prob:.3f}</span>'
                f'<span style="background: {badge_color}; color: white; padding: 0.1rem 0.5rem; border-radius: 3px; font-size: 0.8rem;">{badge}</span>'
                f'</div>'
                f'<div style="margin: 0.3rem 0; font-size: 1.05rem;">'
                f'{src} <span style="font-size: 0.75rem; color: #888;">({st_src_t})</span>'
                f' &rarr; {tgt} <span style="font-size: 0.75rem; color: #888;">({st_tgt_t})</span>'
                f'</div>'
                f'<div style="font-size: 0.85rem; color: #bbb;">{rationale}</div>'
                f'<div style="font-size: 0.85rem; color: #bbb; margin-top: 0.15rem;">{note}</div>'
                f'<div style="font-size: 0.85rem; color: #2ecc71; margin-top: 0.2rem;">→ {action}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Side-by-side claim comparison
            s1, s2, s3 = st.columns(3)
            s1.metric(f"{src} claims reachable", src_cnt, help="How many claims/perceptions this node can already reach")
            s2.metric(f"{tgt} claims reachable", tgt_cnt, help="How many claims/perceptions the target can already reach")
            s3.metric("New pathways this link creates", pathways, help=f"New narrative paths opened: {new_vds}")

            with st.expander("Which specific claims it touches", expanded=False):
                new_nodes_raw = row.get("new_pathway_nodes", "")
                if new_nodes_raw and new_nodes_raw != "none":
                    c_have, c_new, c_gain = st.columns(3)
                    with c_have:
                        st.markdown(f"**{src} currently has**")
                    with c_new:
                        st.markdown("**New connections the link creates**")
                    with c_gain:
                        st.markdown(f"**{tgt} would gain**")

                    for part in new_nodes_raw.split(";"):
                        part = part.strip()
                        if not part:
                            continue
                        if part.startswith("+") or part == "more":
                            continue
                        vd_tag = f"`{part[part.rfind('['):].strip('[]')}`" if "[" in part and "]" in part else ""
                        claim_text = part[:part.rfind("[")].strip() if "[" in part else part
                        if "saw:" in part or "have:" in part:
                            with c_have:
                                st.markdown(f"- {claim_text.replace('saw:', '').replace('have:', '').strip()} {vd_tag}")
                        elif "lacks:" in part:
                            with c_new:
                                st.markdown(f"- {claim_text.replace('lacks:', '').strip()} {vd_tag}")
                        elif "getting:" in part or "helps:" in part:
                            with c_gain:
                                st.markdown(f"- {claim_text.replace('getting:', '').replace('helps:', '').strip()} {vd_tag}")
                        else:
                            with c_new:
                                st.markdown(f"- {claim_text} {vd_tag}")
                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**{src}** value dimensions: {src_vds}")
                with c2:
                    st.markdown(f"**{tgt}** value dimensions: {tgt_vds}")

            st.divider()

    # ── 3. Suggested Projects ──
    if not prototype_candidates.empty:
        st.markdown(f"**3. Suggested projects to invest in**")
        st.caption(
            "These projects scored highest as prototype investment candidates. "
            "The **prototype candidate score** combines: network centrality (how well-connected), "
            "narrative reach (how many claims/perceptions it connects to), and budget efficiency "
            "(lower budget with higher reach = better value). "
            "The **budget** comes from the synthetic financial dataset "
            "(`04_investment_opportunity_synthetic.py`) which maps each project's annual spending "
            "based on KTool's financial bridge data."
        )

        top_n = min(10, len(prototype_candidates))
        proj_view = prototype_candidates.sort_values("prototype_candidate_score", ascending=False).head(top_n)
        show_cols = [c for c in ["label", "associated_budget", "prototype_candidate_score", "suggestion_reason", "degree", "impact_level"] if c in proj_view.columns]
        proj_display = proj_view[show_cols].copy()
        proj_display.columns = [c.replace("_", " ").title() for c in show_cols]
        if "Associated Budget" in proj_display.columns:
            proj_display["Associated Budget"] = proj_display["Associated Budget"].apply(
                lambda b: f"€{b:,.0f}" if pd.notna(b) and b > 0 else "—"
            )
        if "Degree" in proj_display.columns:
            proj_display["Degree"] = proj_display["Degree"].astype(int)
        if "Prototype Candidate Score" in proj_display.columns:
            proj_display["Prototype Candidate Score"] = proj_display["Prototype Candidate Score"].apply(lambda v: f"{v:.3f}")
        st.dataframe(proj_display, width='stretch', hide_index=True)

    # ── 4. Value Dimension Landscape ──
    if impact_dashboard:
        vd_list = impact_dashboard.get("vd_summary", [])
        if vd_list:
            st.divider()
            st.markdown("**4. Value dimension landscape**")
            st.caption(
                "Claim count per value dimension. "
                "Budget = total investment (€) mapped from the financial bridge "
                "(`04_investment_opportunity_synthetic.py`). "
                "Underfunded = dimensions with the least budget allocation."
            )
            vd_df = pd.DataFrame(vd_list)
            show_vd_cols = {"value_dimension": "Value dimension", "claim_count": "Claims"}
            has_budget = vd_df["budget"].sum() > 0 if "budget" in vd_df.columns else False
            has_under = vd_df["underfunded"].any() if "underfunded" in vd_df.columns else False
            if has_budget:
                vd_df["budget_display"] = vd_df["budget"].apply(lambda b: f"€{b:,.0f}")
                show_vd_cols["budget_display"] = "Budget (€)"
            if has_under:
                show_vd_cols["underfunded"] = "Underfunded"
            col_config = {}
            if has_under:
                col_config["underfunded"] = st.column_config.CheckboxColumn("Underfunded")
            st.dataframe(
                vd_df.rename(columns=show_vd_cols)[list(show_vd_cols.values())],
                column_config=col_config,
                width='stretch', hide_index=True,
            )

    if not link_intervention_scores.empty:
        st.divider()
        st.markdown("**7. Full simulation results**")
        st.caption("What changes when we add each proposed link.")

        if link_intervention_summary:
            m1, m2, m3 = st.columns(3)
            m1.metric("Simulated links", int(link_intervention_summary.get("n_recommendations_simulated", 0)))
            m2.metric("Merge clusters", int(link_intervention_summary.get("n_merging_components", 0)),
                      help="How many proposed links connect two separate clusters.")
            m3.metric("Avg sensitivity", f"{link_intervention_summary.get('avg_sensitivity', 0.0):.3f}",
                      help="Higher = this link would change the conversation flow more.")

        view = link_intervention_scores.copy()
        show_cols = [c for c in [
            "source_label", "target_label", "link_type",
            "merges_components", "sensitivity_score",
            "avg_perception_PR_delta", "n_perceptions_gaining", "n_perceptions_losing",
            "max_perception_PR_delta", "info_reach_expansion",
        ] if c in view.columns]
        if show_cols:
            st.dataframe(
                view[show_cols].rename(columns={
                    "source_label": "From", "target_label": "To", "link_type": "Link type",
                    "merges_components": "Merges components", "sensitivity_score": "Sensitivity",
                    "avg_perception_PR_delta": "Avg PR delta (perceptions)",
                    "n_perceptions_gaining": "Perceptions gaining PR",
                    "n_perceptions_losing": "Perceptions losing PR",
                    "max_perception_PR_delta": "Max PR delta (any perception)",
                    "info_reach_expansion": "New info nodes reachable",
                }),
                width='stretch', hide_index=True,
            )

        with st.expander("How sensitivity is calculated"):
            st.markdown(
                "- **Sensitivity** = combines influence shift, cluster merging, and how many new items become reachable.\n"
                "- **Influence shift** = how much attention flows change when the link is added.\n"
                "- **Merges clusters** = link connects two previously separate parts of the network.\n"
            )

    if not gnn_summary:
        st.info(
            "No GNN summary found yet. Run:\n\n"
            "```\npython src/analysis/11_gnn_preparation.py\n```"
        )
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Nodes", f"{int(gnn_summary.get('node_count', 0)):,}")
        m2.metric("Edges", f"{int(gnn_summary.get('edge_count', 0)):,}")
        m3.metric("Relations", int(gnn_summary.get('relation_count', 0)))
        m4.metric("Features per node", int(gnn_summary.get('feature_dim', 0)))

        c1, c2 = st.columns([0.58, 0.42])
        with c1:
            semantic_coverage = float(gnn_summary.get('semantic_embedding_coverage_rate', 0.0))
            semantic_source = gnn_summary.get('semantic_embedding_source') or "None"
            summary_items = {
                "Semantic coverage": f"{semantic_coverage:.1%}",
                "Low-confidence edges": int(gnn_summary.get('low_confidence_edge_count', 0)),
                "Semantic source": semantic_source,
            }
            st.markdown("**Model setup**")
            summary_df = pd.DataFrame(list(summary_items.items()), columns=["Metric", "Value"])
            summary_df["Value"] = summary_df["Value"].astype(str)
            st.dataframe(summary_df, width='stretch', hide_index=True)

        with c2:
            st.markdown("**Pipeline steps**")
            for stage in gnn_summary.get("pipeline_stages", []):
                st.write(f"- {stage}")

        if gnn_training_report:
            st.divider()
            st.markdown("**Latest training results**")
            metrics = gnn_training_report.get("metrics", {})
            train_metrics = metrics.get("train", {})
            val_metrics = metrics.get("validation", {})
            test_metrics = metrics.get("test", {})
            t1, t2, t3 = st.columns(3)
            t1.metric("Train accuracy", f"{float(train_metrics.get('accuracy', 0.0)):.1%}")
            t2.metric("Validation accuracy", f"{float(val_metrics.get('accuracy', 0.0)):.1%}")
            t3.metric("Test accuracy", f"{float(test_metrics.get('accuracy', 0.0)):.1%}")

            st.caption(
                f"R-GCN model: {gnn_training_report.get('model', 'unknown')} · "
                f"Classes: {gnn_training_report.get('class_count', 0)} · "
                f"Relations: {gnn_training_report.get('relation_count', 0)}"
            )

        if gnn_link_report:
            st.divider()
            st.markdown("**Link prediction results**")
            link_metrics = gnn_link_report.get("metrics", {})
            link_train = link_metrics.get("train", {})
            link_val = link_metrics.get("validation", {})
            link_test = link_metrics.get("test", {})
            l1, l2, l3 = st.columns(3)
            l1.metric("Train AUC", f"{float(link_train.get('auc', 0.0)):.1%}")
            l2.metric("Validation AUC", f"{float(link_val.get('auc', 0.0)):.1%}")
            l3.metric("Test AUC", f"{float(link_test.get('auc', 0.0)):.1%}")

            ap1, ap2, ap3 = st.columns(3)
            ap1.metric("Train AP", f"{float(link_train.get('average_precision', 0.0)):.1%}")
            ap2.metric("Validation AP", f"{float(link_val.get('average_precision', 0.0)):.1%}")
            ap3.metric("Test AP", f"{float(link_test.get('average_precision', 0.0)):.1%}")

            st.caption(
                f"Link predictor: {gnn_link_report.get('model', 'unknown')} · "
                f"Filtered edges: {gnn_link_report.get('edge_count_after_filter', 0)} · "
                f"Best validation AUC: {float(gnn_link_report.get('best_validation_auc', 0.0)):.1%}"
            )

    # ── Diagnostics & Limitations ──
    with st.expander("GNN Diagnostics & Limitations", expanded=False):
        st.markdown(
            "### What the GNN does well\n"
            "- Detects **structural holes** — missing links that graph topology suggests should exist\n"
            "- Scores candidates by **narrative impact** (new pathways, value dimensions unlocked)\n"
            "- Applies a **domain compatibility filter** (embedding cosine similarity) to penalise cross-domain links\n\n"
            "### Known limitations\n"
            "- **53 / 375 nodes have zero embeddings** — their semantic content is unrecoverable, so domain filter falls back to node_type heuristic\n"
            "- **Topological false positives** — structurally similar agents in different operational domains (e.g., Focus Ireland ↔ Liquid Therapy) get high GNN scores but low governance plausibility\n"
            "- **No perception/claim awareness at training time** — the R-GCN uses only mapping-layer features; narrative impact is computed post-hoc\n"
            "- **Sparse training edges** — only 8 training examples per relation type\n\n"
            "### Recommendations with domain compatibility\n"
        )

        if not impact_edge_candidates.empty:
            dc_view = impact_edge_candidates[
                [c for c in ["source_label", "target_label", "bridge_type",
                             "domain_compatibility", "domain_note",
                             "composite_governance_score", "action_template"]
                 if c in impact_edge_candidates.columns]
            ].copy()
            if not dc_view.empty:
                st.dataframe(dc_view, width='stretch', hide_index=True)

        st.markdown(
            "### When to trust / not trust\n\n"
            "| Scenario | Trust | Reason |\n"
            "|---|---|---|\n"
            "| Same value dimension, components merged | ✅ | Measurable new narrative pathways |\n"
            "| Domain cosine sim > 0.5 | ✅ | Same operational domain |\n"
            "| Domain cosine sim 0.3–0.5 | ⚠️ | Structurally plausible, semantically weak |\n"
            "| Zero embedding on either node | ⚠️ | No semantic signal to verify |\n"
            "| Different bridge, cosine sim < 0.3 | ❌ | Likely topological false positive |\n\n"
            "### Diagnostic file\n"
            "See `docs/gnn_diagnostics.md` for a full breakdown of model architecture, data quality "
            "checks, and future improvements.\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Geography & Evolution
# ═══════════════════════════════════════════════════════════════════════════════
    with tab_geo:
        st.subheader("Geography & Ecosystem Evolution")

        geo_col1, geo_col2 = st.columns([0.55, 0.45])

        with geo_col1:
            st.markdown("**Irish ecosystem map**")
            if "location" not in nodes_win.columns:
                nodes_win["location"] = ""
            geo_nodes = nodes_win[nodes_win["location"].notna() & (nodes_win["location"] != "")].copy()
            if "latitude" in geo_nodes.columns and "longitude" in geo_nodes.columns:
                geo_nodes["latitude"] = pd.to_numeric(geo_nodes["latitude"], errors="coerce")
                geo_nodes["longitude"] = pd.to_numeric(geo_nodes["longitude"], errors="coerce")
                geo_plot = geo_nodes.dropna(subset=["latitude", "longitude"])
            else:
                geo_plot = pd.DataFrame()

            if not geo_plot.empty:
                    map_modes = ["Node locations", "Concentration", "Influence", "Underserved areas", "County heatmap", "Time-lapse"]
                    map_mode = st.radio("Map view", map_modes, horizontal=True, label_visibility="collapsed")
                    geo_fig = go.Figure()

                    VD_COLORS = {
                        "social_justice": "#e41a1c",
                        "evidence_based": "#377eb8",
                        "innovation_drive": "#4daf4a",
                        "collaboration": "#984ea3",
                        "cultural_identity": "#ff7f00",
                        "community_autonomy": "#f0f000",
                        "austerity_scarcity": "#a65628",
                    }
                    VD_LABELS = {
                        "social_justice": "Social Justice",
                        "evidence_based": "Evidence-Based",
                        "innovation_drive": "Innovation Drive",
                        "collaboration": "Collaboration",
                        "cultural_identity": "Cultural Identity",
                        "community_autonomy": "Community Autonomy",
                        "austerity_scarcity": "Austerity & Scarcity",
                    }

                    if map_mode == "Node locations":
                        nt_colors = {nt: color_for_node_type(nt) for nt in geo_plot["node_type"].unique()}
                        for nt in geo_plot["node_type"].unique():
                            sub = geo_plot[geo_plot["node_type"] == nt]
                            geo_fig.add_trace(go.Scattermap(
                                lat=sub["latitude"].tolist(), lon=sub["longitude"].tolist(),
                                mode="markers+text", marker=dict(size=10, color=nt_colors[nt], opacity=0.8),
                                text=sub["label"].tolist(), textposition="top center",
                                name=nt.title(),
                                hovertemplate="<b>%{text}</b><br>Type: " + nt.title() + "<br>Location: %{customdata}<br><extra></extra>",
                                customdata=sub["location"].tolist(),
                            ))

                    elif map_mode == "Concentration":
                        loc_counts = geo_plot.groupby("location").agg(
                            count=("global_id", "nunique"),
                            lat=("latitude", "first"), lon=("longitude", "first"),
                        ).reset_index()
                        max_c = loc_counts["count"].max()
                        geo_fig.add_trace(go.Scattermap(
                            lat=loc_counts["lat"].tolist(), lon=loc_counts["lon"].tolist(),
                            mode="markers+text",
                            marker=dict(
                                size=[max(8, 12 + 40 * (c / max_c)) for c in loc_counts["count"]],
                                color=loc_counts["count"], colorscale="Viridis", showscale=True,
                                opacity=0.7, sizemin=8,
                            ),
                            text=loc_counts["location"].tolist(), textposition="top center",
                            hovertemplate="<b>%{text}</b><br>Nodes: %{marker.size}<extra></extra>",
                        ))

                    elif map_mode == "Influence":
                        loc_pr = geo_plot.groupby("location").agg(
                            lat=("latitude", "first"), lon=("longitude", "first"),
                            nodes=("global_id", list),
                        ).reset_index()
                        pr_by_gid = {}
                        if not diffusion.empty:
                            for _, r in diffusion.iterrows():
                                pr_by_gid[str(r.get("global_id", ""))] = float(r.get("pagerank", 0) or 0)
                        loc_pr["total_pr"] = loc_pr["nodes"].apply(lambda ids: sum(pr_by_gid.get(str(i), 0.0) for i in ids))
                        max_pr = loc_pr["total_pr"].max()
                        if max_pr > 0:
                            geo_fig.add_trace(go.Scattermap(
                                lat=loc_pr["lat"].tolist(), lon=loc_pr["lon"].tolist(),
                                mode="markers+text",
                                marker=dict(
                                    size=[max(8, 12 + 40 * (v / max_pr)) for v in loc_pr["total_pr"]],
                                    color=loc_pr["total_pr"], colorscale="Plasma", showscale=True,
                                    opacity=0.7,
                                ),
                                text=loc_pr["location"].tolist(), textposition="top center",
                                hovertemplate="<b>%{text}</b><br>Total influence: %{marker.color:.4f}<extra></extra>",
                            ))
                        else:
                            st.info("PageRank data not available. Run the pipeline first.")

                    elif map_mode == "Underserved areas":
                        loc_types = geo_plot.groupby("location").agg(
                            lat=("latitude", "first"), lon=("longitude", "first"),
                            types=("node_type", lambda x: x.tolist()),
                        ).reset_index()

                        def support_score(types_list):
                            agents = sum(1 for t in types_list if t in ("agent", "channel"))
                            projects = sum(1 for t in types_list if t in ("project", "pilot", "prototype"))
                            total = len(types_list)
                            if total == 0:
                                return 0.0
                            support = agents / max(projects, 1)
                            return min(support, 3.0) / 3.0

                        loc_types["score"] = loc_types["types"].apply(support_score)
                        geo_fig.add_trace(go.Scattermap(
                            lat=loc_types["lat"].tolist(), lon=loc_types["lon"].tolist(),
                            mode="markers+text",
                            marker=dict(
                                size=20,
                                color=loc_types["score"], colorscale="RdYlGn", showscale=True,
                                opacity=0.7, cmin=0, cmax=1,
                            ),
                            text=loc_types["location"].tolist(), textposition="top center",
                            hovertemplate="<b>%{text}</b><br>Support score: %{marker.color:.2f}<br>(1.0 = balanced, 0.0 = underserved)<extra></extra>",
                        ))

                    elif map_mode in ("County heatmap", "Time-lapse"):
                        geo_json_candidates = [
                            ROOT / "data" / "processed" / "ireland_counties.geojson",
                            ANALYSIS_DIR / "ireland_counties.geojson",
                        ]
                        geo_json_path = next((p for p in geo_json_candidates if p.exists()), None)
                        if geo_json_path is None:
                            st.info("County boundary file not found. Re-run the pipeline for the synthetic platform.")
                        else:
                            with open(geo_json_path) as f:
                                ireland_counties = json.load(f)

                            if "value_dimension" not in nodes_win.columns:
                                nodes_win["value_dimension"] = ""
                            has_county = nodes_win[nodes_win["county"].notna() & nodes_win["value_dimension"].notna()].copy()
                            if has_county.empty:
                                st.info("No nodes with both county and value_dimension data. Run `28_enrich_value_dimensions.py` first.")
                            else:
                                if "sim_year" in has_county.columns:
                                    has_county["year"] = pd.to_numeric(has_county["sim_year"], errors="coerce").fillna(has_county["best_date"].dt.year)
                                else:
                                    has_county["year"] = has_county["best_date"].dt.year

                                def build_vd_order(vd_series):
                                    present = sorted(vd_series.unique(), key=lambda v: list(VD_COLORS.keys()).index(v) if v in VD_COLORS else 99)
                                    vd_code = {v: i for i, v in enumerate(present)}
                                    n = len(present)
                                    steps = []
                                    for i, vd in enumerate(present):
                                        lo = i / n
                                        hi = (i + 1) / n
                                        c = VD_COLORS.get(vd, "#888888")
                                        steps.append([lo, c])
                                        steps.append([hi - 0.001, c])
                                        steps.append([hi, c])
                                    return present, vd_code, n, steps

                                def dominant_per_county(vd_year_group):
                                    dom = vd_year_group.loc[vd_year_group.groupby("county")["count"].idxmax()].reset_index(drop=True)
                                    return dom if not dom.empty else pd.DataFrame()

                                def county_centroids(gj_dict):
                                    centroids = {}
                                    for f in gj_dict["features"]:
                                        name = f["properties"].get("county", "").upper()
                                        coords = f["geometry"]["coordinates"]
                                        if f["geometry"]["type"] == "MultiPolygon":
                                            pts = [p for poly in coords for ring in poly for p in ring]
                                        elif f["geometry"]["type"] == "Polygon":
                                            pts = [p for ring in coords for p in ring]
                                        else:
                                            continue
                                        if pts:
                                            centroids[name] = [sum(p[1] for p in pts) / len(pts), sum(p[0] for p in pts) / len(pts)]
                                    return centroids

                                centroids = county_centroids(ireland_counties)
                                vd_legend = " | ".join(f'<span style="display:inline-block;width:10px;height:10px;background:{VD_COLORS[vd]};border-radius:50%;"></span> {VD_LABELS.get(vd, vd.replace("_"," ").title())}' for vd in VD_COLORS)

                                def county_trace(dom_df, marker_size_col, show_legend=False):
                                    dom_df = dom_df[dom_df["county"].str.upper().isin(centroids)].copy()
                                    if dom_df.empty:
                                        return None
                                    dom_df["lat"] = dom_df["county"].str.upper().map(lambda c: centroids.get(c, [None, None])[0])
                                    dom_df["lon"] = dom_df["county"].str.upper().map(lambda c: centroids.get(c, [None, None])[1])
                                    dom_df = dom_df.dropna(subset=["lat", "lon"])
                                    if dom_df.empty:
                                        return None
                                    colors = dom_df["value_dimension"].map(VD_COLORS).tolist()
                                    labels = dom_df["value_dimension"].map(VD_LABELS).tolist()
                                    return go.Scattermap(
                                        lat=dom_df["lat"].tolist(), lon=dom_df["lon"].tolist(),
                                        mode="markers+text",
                                        marker=dict(
                                            size=dom_df[marker_size_col].tolist() if marker_size_col in dom_df else 18,
                                            color=colors, opacity=0.85,
                                            showscale=False,
                                        ),
                                        text=dom_df["county"].tolist(), textposition="top center",
                                        hovertemplate="<b>%{text}</b><br>%{customdata}<br>Nodes: %{marker.size}<extra></extra>",
                                        customdata=labels,
                                        showlegend=show_legend,
                                    )

                                if map_mode == "County heatmap":
                                    county_vd = has_county.groupby(["county", "value_dimension"]).size().reset_index(name="count")
                                    dominant = dominant_per_county(county_vd)
                                    tr = county_trace(dominant, "count")
                                    if tr is not None:
                                        geo_fig.add_trace(tr)
                                    else:
                                        st.info("No county centroids matched. Check GeoJSON county names.")
                                    st.caption(f"County centroid markers — size = node count, color = dominant value dimension. ROI only.<br>{vd_legend}", unsafe_allow_html=True)

                                else:
                                    has_date = has_county[has_county["year"].notna()].copy()
                                    years_sorted = sorted(has_date["year"].unique().astype(int))
                                    if len(years_sorted) < 2:
                                        st.info("Only one year of data available. Switch to 'County heatmap' for the static view.")
                                    else:
                                        county_vd_year = has_date.groupby(["year", "county", "value_dimension"]).size().reset_index(name="count")

                                        def make_frame_data(yr):
                                            yr_data = county_vd_year[county_vd_year["year"] == yr]
                                            dom = dominant_per_county(yr_data)
                                            return county_trace(dom, "count")

                                        tr0 = make_frame_data(years_sorted[0])
                                        if tr0 is not None:
                                            geo_fig.add_trace(tr0)
                                        frames = []
                                        for yr in years_sorted[1:]:
                                            tr_f = make_frame_data(yr)
                                            if tr_f is not None:
                                                frames.append(go.Frame(data=[tr_f], name=str(int(yr))))
                                        if frames:
                                            geo_fig.frames = frames
                                            slider_steps = [dict(method="animate", label=str(int(yr)), args=[[str(int(yr))], dict(mode="immediate", transition=dict(duration=300))]) for yr in years_sorted]
                                            geo_fig.update_layout(
                                                updatemenus=[dict(
                                                    type="buttons", showactive=False, x=0.05, y=1.1,
                                                    buttons=[
                                                        dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=800, redraw=True), fromcurrent=True)]),
                                                        dict(label="Pause", method="animate", args=[[None], dict(mode="immediate", transition=dict(duration=0))]),
                                                    ],
                                                )],
                                                sliders=[dict(active=0, currentvalue=dict(prefix="Year: "), pad=dict(t=10), steps=slider_steps)],
                                            )
                                        st.caption(f"County centroid markers over time — color = dominant value dimension. Play the animation.<br>{vd_legend}", unsafe_allow_html=True)

                    geo_fig.update_layout(
                        map=dict(style="open-street-map", center=dict(lat=53.4, lon=-8.0), zoom=6.5),
                        margin=dict(l=0, r=0, t=0, b=0), height=550,
                    )
                    if map_mode == "Node locations":
                        geo_fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(geo_fig, use_container_width=True)
            else:
                st.info(f"No nodes with valid coordinates in time window ({len(geo_nodes)} have location, 0 with valid lat/lon). Run `00_enrich_locations.py` first.")

    with geo_col2:
        st.markdown("**Node type distribution on map**")
        if not geo_nodes.empty:
            type_loc_counts = geo_nodes.groupby(["node_type", "location"]).size().reset_index(name="count")
            st.dataframe(type_loc_counts, width='stretch', hide_index=True)

        st.divider()
        st.markdown("**Ecosystem growth over time**")
        growth = nodes[nodes["best_date"].notna()].copy()
        growth["year_month"] = growth["best_date"].dt.to_period("M").astype(str)
        growth_ts = growth.groupby(["year_month", "node_type"]).size().reset_index(name="count")
        growth_cumul = growth_ts.copy()
        growth_cumul["cumulative"] = growth_cumul.groupby("node_type")["count"].cumsum()
        fig_growth = px.line(
            growth_cumul, x="year_month", y="cumulative", color="node_type",
            template="plotly_white",
            title="Cumulative node count by type over time",
            color_discrete_sequence=PLOTLY_PALETTE,
        )
        fig_growth.update_layout(height=280, margin=dict(l=10, r=10, t=42, b=10))
        st.plotly_chart(fig_growth, use_container_width=True)

        st.markdown("**Time window composition**")
        type_counts_now = nodes_win["node_type"].value_counts().reset_index()
        type_counts_now.columns = ["Node type", "Count"]
        st.dataframe(type_counts_now, width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Network Layers
# ═══════════════════════════════════════════════════════════════════════════════
with tab_layer:
    st.subheader("What each link type adds")

    layer_metrics = readiness.get("layer_family_analysis", {})
    if layer_metrics:
        layer_df = pd.DataFrame([
            {
                "Link type": friendly_edge_family(layer),
                "Connections": vals.get("edge_count", 0),
                "Clusters": vals.get("component_count", 0),
                "Density": vals.get("graph_density", 0.0),
            }
            for layer, vals in layer_metrics.items()
        ])
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(layer_df, x="Link type", y="Connections",
                                   template="plotly_white", title="Links by type"), width='stretch')
        with c2:
            st.plotly_chart(px.bar(layer_df, x="Link type", y="Clusters",
                                   template="plotly_white", title="Clusters by link type"), width='stretch')
    else:
        st.info("Link type report not available.")

    st.markdown("**Filter: show/hide link types**")
    family_source_column = "edge_family_label" if "edge_family_label" in edges.columns else "edge_family"
    available_families_raw = sorted(edges[family_source_column].dropna().astype(str).unique().tolist()) if family_source_column in edges.columns else []
    family_display_map = {k: friendly_edge_family(k) for k in available_families_raw}
    available_family_display = list(family_display_map.values())
    selected_families = st.multiselect(
        "Link types to show",
        options=available_family_display,
        default=available_family_display,
        help="Uncheck a type to hide those links and see how the network changes.",
    )
    selected_keys = [k for k, v in family_display_map.items() if v in selected_families]
    filtered_edges = edges[edges[family_source_column].astype(str).isin(selected_keys)].copy() if selected_keys and family_source_column in edges.columns else edges.copy()

    graph = nx.Graph()
    graph.add_nodes_from(nodes["global_id"].astype(str).tolist())
    for _, row in filtered_edges.iterrows():
        graph.add_edge(str(row["source_global_id"]), str(row["target_global_id"]))
    component_count = nx.number_connected_components(graph) if graph.number_of_nodes() else 0
    largest_size = len(max(nx.connected_components(graph), key=len)) if component_count else 0
    largest_share = largest_size / graph.number_of_nodes() if graph.number_of_nodes() else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Links shown", int(filtered_edges.shape[0]))
    m2.metric("Clusters", int(component_count))
    m3.metric("Biggest cluster", f"{largest_share:.1%}")
    st.caption("If a link type reduces clusters and grows the biggest one, it connects previously separate groups.")

    with st.expander("Where each link type comes from"):
        st.markdown(
            "| Link type | What it records | Where from |\n"
            "| --- | --- | --- |\n"
            "| **Declared links** | Agent↔project, project↔perception, initiative interconnections | Source database |\n"
            "| **AI-inferred** | Quote-to-quote links (similar, contradictory, causal, sequential) | AI analysis of quote text |\n"
            "| **Listening** | Channel→information→value chains | Data pipeline |\n"
            "| **Narrative analysis** | Perception↔challenge, pattern↔perception links | Source database |\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — AI-Generated Links
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ai_semantic:
    st.subheader("AI-generated links")
    if ai_mode == "Source only":
        st.info("AI toggle set to **Source only** — switch to **All data** in the sidebar to see AI-inferred edges.")
    st.caption("Links the AI found between quotes. Separated from the source data so you can see what's AI vs what's from the original dataset.")

    ai_edges = edges.copy()
    ai_mask = pd.Series([False] * len(ai_edges), index=ai_edges.index)
    if "is_ai_generated" in ai_edges.columns:
        ai_mask = ai_edges["is_ai_generated"].apply(normalize_bool)
    if "edge_origin" in ai_edges.columns:
        ai_mask = ai_mask | ai_edges["edge_origin"].astype(str).str.lower().str.contains("ai|semantic|inference", regex=True)
    ai_edges = ai_edges[ai_mask].copy()

    if ai_edges.empty:
        st.info("No AI links found. Run the semantic edge pipeline first.")
    else:
        ai_total = len(ai_edges)
        ai_share = ai_total / len(edges) if len(edges) else 0
        type_col = "edge_type" if "edge_type" in ai_edges.columns else None

        m1, m2, m3 = st.columns(3)
        m1.metric("AI links", ai_total)
        m2.metric("Share of all links", f"{ai_share:.1%}")
        if type_col:
            m3.metric("Link types", int(ai_edges[type_col].nunique()))

        c1, c2 = st.columns(2)
        with c1:
            if type_col:
                tc = ai_edges[type_col].fillna("unknown").astype(str).value_counts().reset_index()
                tc.columns = ["Type", "Count"]
                fig_t = px.bar(tc, x="Count", y="Type", orientation="h", template="plotly_white",
                               title="AI links by semantic type", color="Type",
                               color_discrete_sequence=PLOTLY_PALETTE)
                fig_t.update_layout(height=360, showlegend=False)
                st.plotly_chart(fig_t, width='stretch')
        with c2:
            param_col = next((c for c in ["alc_semantic_parameter", "parameter"] if c in ai_edges.columns), None)
            if param_col:
                pc = ai_edges[param_col].fillna("unknown").astype(str).value_counts().reset_index()
                pc.columns = ["Parameter", "Count"]
                fig_p = px.bar(pc.head(12), x="Count", y="Parameter", orientation="h",
                               template="plotly_white", title="AI semantic parameters")
                fig_p.update_layout(height=360)
                st.plotly_chart(fig_p, width='stretch')

        node_label_map = {}
        if "global_id" in nodes.columns:
            lbl_series = nodes.apply(lambda r: fallback_label(r.get("label"), r.get("global_id")), axis=1) if "label" in nodes.columns else nodes["global_id"].astype(str)
            node_label_map = dict(zip(nodes["global_id"].astype(str), lbl_series.astype(str)))

        ai_view = ai_edges.copy()
        if "source_global_id" in ai_view.columns:
            ai_view["Source"] = ai_view["source_global_id"].astype(str).map(node_label_map).fillna(ai_view["source_global_id"].astype(str))
        if "target_global_id" in ai_view.columns:
            ai_view["Target"] = ai_view["target_global_id"].astype(str).map(node_label_map).fillna(ai_view["target_global_id"].astype(str))

        type_filter_opts = sorted(ai_view[type_col].dropna().astype(str).unique().tolist()) if type_col else []
        selected_types = st.multiselect("Filter semantic types", type_filter_opts, default=type_filter_opts)
        if selected_types and type_col:
            ai_view = ai_view[ai_view[type_col].astype(str).isin(selected_types)]

        show_cols = [c for c in ["Source", "Target", "edge_type", "alc_semantic_parameter",
                                 "inference_method", "generated_by", "description"] if c in ai_view.columns]
        if show_cols:
            st.dataframe(
                ai_view[show_cols].head(80).rename(columns={
                    "edge_type": "Semantic type", "alc_semantic_parameter": "Parameter",
                    "inference_method": "Method", "generated_by": "Generated by", "description": "Explanation",
                }),
                width='stretch', hide_index=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 14 — Budget & Finance (synthetic only, appended last)
# ═══════════════════════════════════════════════════════════════════════════════
if tab_financial is not None:
    with tab_financial:
        st.subheader("Budget & Finance Analysis")
        st.caption(
            "Synthetic dataset: same network as Platform 173, but with fake budgets added "
            "so we can test investment scenarios."
        )

        if leverage_df.empty and stranded_df.empty and fin_diffusion_df.empty:
            st.warning(
                "Financial analysis data not found. Run:\n\n"
                "```\nset KTOOL_PLATFORM_ID=173_synthetic\n"
                "python src/analysis/04_investment_opportunity_synthetic.py\n```"
            )
        else:
            if fin_summary:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Items with budget", fin_summary.get("n_initiatives_with_budget", "—"))
                m2.metric("Agents with investment", fin_summary.get("n_agents_with_investment", "—"))
                m3.metric("Stranded assets", fin_summary.get("stranded_assets_count", "—"),
                          help="Items with budget ≥ €500k but few connections and low bridge score.")
                if not leverage_df.empty and "leverage_score" in leverage_df.columns:
                    top_leverage = leverage_df["leverage_score"].max()
                    m4.metric("Top leverage score", f"{top_leverage:.3f}",
                              help="Bridge score per €1M budget — higher = more impact per euro.")

            st.divider()

            if not leverage_df.empty:
                st.markdown("### A. Value for money — budget vs bridge score")
                st.caption("Items that connect many groups and don't cost much.")
                if "associated_budget" in leverage_df.columns and "betweenness_centrality" in leverage_df.columns:
                    lev_plot = leverage_df.copy()
                    lev_plot["label"] = lev_plot.get("label", lev_plot["global_id"]).astype(str).str[:35]
                    lev_plot["budget_M"] = lev_plot["associated_budget"] / 1_000_000
                    color_col = "investment_level" if "investment_level" in lev_plot.columns else "node_type"
                    fig_lev = px.scatter(
                        lev_plot, x="budget_M", y="betweenness_centrality",
                        color=color_col, size="leverage_score", hover_name="label",
                        template="plotly_white",
                        title="Budget vs network bridge score",
                        labels={"budget_M": "Budget (€M)", "betweenness_centrality": "Bridge score"},
                        color_discrete_sequence=PLOTLY_PALETTE,
                    )
                    fig_lev.update_layout(height=480, legend_title=color_col.replace("_", " ").title())
                    st.plotly_chart(fig_lev, width='stretch')
                    st.markdown("**Top 10 best value items**")
                    show_cols = [c for c in ["label", "node_type", "investment_level", "associated_budget",
                                             "betweenness_centrality", "leverage_score"] if c in leverage_df.columns]
                    st.dataframe(
                        leverage_df[show_cols].head(10).rename(columns={
                            "label": "Item", "node_type": "Type",
                            "investment_level": "Budget tier", "associated_budget": "Budget (€)",
                            "betweenness_centrality": "Bridge score", "leverage_score": "Leverage",
                        }), width='stretch', hide_index=True,
                    )

            st.divider()

            if not stranded_df.empty:
                st.markdown("### B. Stranded assets")
                st.caption("Big budgets, but barely connected to the network.")
                st.error(f"{len(stranded_df)} stranded assets detected")
                show_cols = [c for c in ["label", "node_type", "associated_budget", "betweenness_centrality",
                                         "degree", "stranded_reason"] if c in stranded_df.columns]
                st.dataframe(stranded_df[show_cols].rename(columns={
                    "label": "Item", "node_type": "Type", "associated_budget": "Budget (€)",
                    "betweenness_centrality": "Bridge score", "degree": "Connections", "stranded_reason": "Flag",
                }), width='stretch', hide_index=True)
            else:
                st.success("No stranded assets found.")

            st.divider()

            if not fin_diffusion_df.empty and "financial_bias" in fin_diffusion_df.columns:
                st.markdown("### C. Where the money flows")
                st.caption(
                    "Positive = this item gets more attention when weighted by budget. "
                    "Negative = under-reached by financial flows."
                )
                fig_fd = px.histogram(
                    fin_diffusion_df, x="financial_bias", color="node_type",
                    nbins=30, barmode="overlay", opacity=0.6,
                    template="plotly_white",
                    title="Money-weighted attention by type",
                    labels={"financial_bias": "Bias (+ = gets more attention with money weight)"},
                    color_discrete_sequence=PLOTLY_PALETTE,
                )
                fig_fd.add_vline(x=0, line_dash="dash", line_color="grey")
                fig_fd.update_layout(height=380, bargap=0.05)
                st.plotly_chart(fig_fd, width='stretch')
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Gets more budget-attention than expected (bias > 0.3)**")
                    top_fin = fin_diffusion_df[fin_diffusion_df["financial_bias"] > 0.3].nlargest(8, "financial_bias")
                    if not top_fin.empty:
                        show = [c for c in ["label", "node_type", "financial_bias"] if c in top_fin.columns]
                        st.dataframe(top_fin[show].rename(columns={"label": "Entity", "node_type": "Type", "financial_bias": "Bias"}),
                                     width='stretch', hide_index=True)
                    else:
                        st.info("No strongly over-reached nodes.")
                with col_b:
                    st.markdown("**Gets less budget-attention than expected (bias < −0.3)**")
                    bot_fin = fin_diffusion_df[fin_diffusion_df["financial_bias"] < -0.3].nsmallest(8, "financial_bias")
                    if not bot_fin.empty:
                        show = [c for c in ["label", "node_type", "financial_bias"] if c in bot_fin.columns]
                        st.dataframe(bot_fin[show].rename(columns={"label": "Entity", "node_type": "Type", "financial_bias": "Bias"}),
                                     width='stretch', hide_index=True)
                    else:
                        st.info("No strongly under-reached nodes.")

            st.divider()

            st.markdown("### D. Budget reallocation test")
            st.caption("We shuffled budgets 1,000 times to see if the real allocation puts money on central nodes better than random chance.")
            realloc = fin_summary.get("budget_reallocation", {}) if fin_summary else {}
            if realloc:
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean leverage score", f"{realloc.get('observed_mean_leverage', 0):.5f}",
                          delta=f"p={realloc.get('mean_leverage_percentile', 0):.0f}%", delta_color="inverse",
                          help="Lower percentile = budgets sit on less central nodes than random.")
                c2.metric("Inequality (Gini)", f"{realloc.get('observed_gini', 0):.3f}",
                          delta=f"p={realloc.get('gini_percentile', 0):.0f}%", delta_color="inverse",
                          help="Higher = more unequal distribution than random.")
                c3.metric("Stranded assets", f"{realloc.get('observed_stranded', 0)}",
                          delta=f"p={realloc.get('stranded_percentile', 0):.0f}%", delta_color="inverse",
                          help=f"Average in simulations: {realloc.get('sim_mean_stranded', 0):.1f}")
                st.error(
                    f"**Result: {realloc.get('efficiency_verdict', 'N/A').replace('_', ' ').title()}** — "
                    f"the current allocation is at the **{realloc.get('mean_leverage_percentile', 0):.0f}th percentile** "
                    f"({realloc.get('n_simulations', 0)} simulations). "
                )
                sim_plot_path = ANALYSIS_DIR / "financial_plots" / "budget_reallocation_simulation.png"
                if sim_plot_path.exists():
                    st.image(str(sim_plot_path), width='stretch')
                with st.expander("How the test works"):
                    st.markdown(
                        "- Budgets get shuffled randomly across items (same amounts, different owners).\n"
                        "- We track: average bridge score, inequality, stranded count.\n"
                        "- p < 5% or p > 95% = real allocation is not random.\n"
                    )

            st.divider()

            st.markdown("### E. Budget by story cluster")
            st.caption("Which story clusters are linked to which budgets, via shared topics.")
            if not quote_clusters.empty and not leverage_df.empty:
                effort_data = []
                for _, row in quote_clusters.iterrows():
                    cluster_id = row.get("cluster_id")
                    topics = [t.strip() for t in str(row.get("topics_thematic_areas", "")).split(";") if t.strip()]
                    for t in topics:
                        matched = leverage_df[
                            leverage_df["label"].str.contains(t.split("|")[0].strip(), case=False, na=False) |
                            leverage_df["label"].str.contains(t, case=False, na=False)]
                        for _, ir in matched.iterrows():
                            effort_data.append({"cluster_id": cluster_id, "topic": t,
                                "initiative": ir.get("label", ""), "budget": ir.get("associated_budget", 0),
                                "leverage": ir.get("leverage_score", 0), "bridge": ir.get("betweenness_centrality", 0)})
                if effort_data:
                    effort_df = pd.DataFrame(effort_data)
                    profile_fin = effort_df.groupby("cluster_id").agg(
                        n_initiatives=("initiative", "nunique"), total_budget=("budget", "sum"),
                        mean_leverage=("leverage", "mean"), mean_bridge=("bridge", "mean"),
                    ).reset_index().sort_values("total_budget", ascending=False)
                    cluster_labels = dict(zip(narrative_profiles["cluster_id"], narrative_profiles["cluster_label"])) if not narrative_profiles.empty else {}
                    profile_fin["cluster"] = profile_fin["cluster_id"].map(cluster_labels).fillna(profile_fin["cluster_id"])
                    fig_ne = px.bar(profile_fin, x="cluster", y="total_budget",
                        color="mean_leverage", color_continuous_scale="RdYlGn",
                        template="plotly_white", title="Budget linked to each story cluster",
                        labels={"cluster": "Story cluster", "total_budget": "Linked budget (€)", "mean_leverage": "Avg leverage"})
                    fig_ne.update_layout(height=350)
                    st.plotly_chart(fig_ne, width='stretch')
                    show_cols = ["cluster", "n_initiatives", "total_budget", "mean_leverage", "mean_bridge"]
                    st.dataframe(profile_fin[show_cols].rename(columns={
                        "cluster": "Story cluster", "n_initiatives": "Items", "total_budget": "Budget (€)",
                        "mean_leverage": "Avg leverage", "mean_bridge": "Avg bridge"}), width='stretch', hide_index=True)
                    st.caption("Topic matching is fuzzy — approximate, not exact.")
                else:
                    st.info("No topic-matched items found.")
            else:
                st.info("Run the narrative pipeline and financial analysis first.")

            st.divider()

            if not link_intervention_scores.empty:
                st.markdown("### F. Link impact sensitivity")
                st.caption("What happens if we add each proposed link.")
                if link_intervention_summary:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Simulated links", int(link_intervention_summary.get("n_recommendations_simulated", 0)))
                    m2.metric("Merge components", int(link_intervention_summary.get("n_merging_components", 0)))
                    m3.metric("Avg sensitivity", f"{link_intervention_summary.get('avg_sensitivity', 0.0):.3f}")
                view = link_intervention_scores.copy()
                show_cols = [c for c in ["source_label", "target_label", "link_type", "merges_components",
                                         "sensitivity_score", "avg_perception_PR_delta"] if c in view.columns]
                if show_cols:
                    st.dataframe(view[show_cols].rename(columns={
                        "source_label": "From", "target_label": "To", "link_type": "Type",
                        "merges_components": "Merges", "sensitivity_score": "Sensitivity",
                        "avg_perception_PR_delta": "Avg PR delta"}), width='stretch', hide_index=True)
                with st.expander("How sensitivity is calculated"):
                    st.markdown("- Sensitivity = `|avg PR delta| × 10 + 0.3 if merges components + reach_expansion / N_info + min(|max PR delta| × 50, 0.2)`")

            st.info(
                "**Bottom line:**\n\n"
                "This test asks: *does money follow network importance?* "
                "Stranded assets (high budget, low connections) suggest it doesn't. "
                "The reallocation simulation confirms: the real budget spread is **worse than random** "
                "(at the 1st percentile) — almost certainly not maximising impact."
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 12 — Narrative Simulation (cluster-level)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_narrative_sim:
    st.subheader("Narrative Simulation — Cluster Opinion Dynamics")
    st.caption(
        "Each story cluster holds an opinion profile (7 value dimensions) aggregated from its claims. "
        "Clusters influence each other through semantic edges between their quotes. "
        "Interventions test what happens when an external agent connects to a cluster."
    )

    if not cluster_narrative_baseline.empty and not cluster_narrative_interventions.empty:
        vd_labels = [
            "Cultural Identity", "Social Justice", "Collaboration",
            "Innovation Drive", "Evidence-Based", "Community Autonomy",
            "Austerity/Scarcity",
        ]
        vd_keys = VD_ORDER

        sim_intro, sim_controls = st.columns([0.6, 0.4])
        with sim_controls:
            target_options = sorted(cluster_narrative_baseline["cluster_id"].unique())
            sel_target = st.selectbox("Target cluster for intervention", target_options, key="cluster_target")
            agent_types = ["Neutral agent", "Funder (collaboration)", "Advocate (social_justice)", "Traditionalist (cultural_identity)"]
            agent_profiles = {
                "Neutral agent": [1/7]*7,
                "Funder (collaboration)": [0.05, 0.05, 0.80, 0.05, 0.05, 0.0, 0.0],
                "Advocate (social_justice)": [0.05, 0.80, 0.05, 0.05, 0.05, 0.0, 0.0],
                "Traditionalist (cultural_identity)": [0.80, 0.05, 0.05, 0.05, 0.0, 0.0, 0.05],
            }
            sel_agent = st.selectbox("Agent type", agent_types, key="agent_type")

        with sim_intro:
            st.markdown(
                "**How it works** — Friedkin-Johnsen anchoring:\n\n"
                "1. Each cluster has a **baseline opinion** = normalized value-dimension profile from its claims.\n"
                "2. Clusters influence each other proportionally to cross-cluster semantic edge weight.\n"
                "3. **Stubbornness** (lambda) scales with claim count: more claims = more anchored.\n"
                "4. Intervention: an external agent with a given opinion profile connects to the target cluster. "
                "The system re-converges; deltas show how opinions shift."
            )

        st.divider()

        # Baseline radar
        col_radar, col_deltas = st.columns([0.6, 0.4])

        with col_radar:
            with st.expander("Cluster opinion profiles (baseline)", expanded=True):
                fig_radar = go.Figure()
                for _, r in cluster_narrative_baseline.iterrows():
                    cid = r["cluster_id"]
                    vals = [float(r.get(k, 0)) for k in vd_keys]
                    label = f"{cid} ({int(r.get('claim_count', 0))} claims, \u03bb={r.get('stubbornness', 0):.2f})"
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=vd_labels + [vd_labels[0]],
                        fill=None,
                        name=label,
                        line=dict(width=1),
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    height=450, margin=dict(l=80, r=80, t=40, b=40),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        with col_deltas:
            st.markdown("**Intervention effect**")
            sub = cluster_narrative_interventions[
                (cluster_narrative_interventions["target_cluster"] == sel_target)
            ].copy()
            if not sub.empty:
                delta_cols = [f"delta_{k}" for k in vd_keys]
                sub["max_delta"] = sub[delta_cols].abs().max(axis=1)
                sub = sub.sort_values("max_delta", ascending=False)
                for _, r in sub.iterrows():
                    cid = r["affected_cluster"]
                    md = r["max_delta"]
                    arrow = "▲" if md > 0 else "▼"
                    st.metric(f"{cid}", f"{arrow} {md:.6f}", help="Max absolute opinion shift across all dimensions")
            else:
                st.info("Select a target cluster to see effects.")

        st.divider()

        # Cluster adjacency network
        with st.expander("Cluster influence network", expanded=False):
            if not cluster_narrative_adjacency.empty:
                G_cluster = nx.Graph()
                for _, r in cluster_narrative_adjacency.iterrows():
                    G_cluster.add_edge(r["source_cluster"], r["target_cluster"], weight=r["weight"])
                pos = nx.spring_layout(G_cluster, seed=42)
                edge_trace = go.Scatter(
                    x=[], y=[], mode="lines", line=dict(width=1, color="#ccc"), hoverinfo="none"
                )
                for u, v, d in G_cluster.edges(data=True):
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_trace["x"] += (x0, x1, None)
                    edge_trace["y"] += (y0, y1, None)
                node_trace = go.Scatter(
                    x=[], y=[], mode="markers+text", text=[], hoverinfo="text",
                    marker=dict(size=20, color="#2ecc71", line=dict(width=2, color="white")),
                )
                for n in G_cluster.nodes():
                    x, y = pos[n]
                    node_trace["x"] += (x,)
                    node_trace["y"] += (y,)
                    node_trace["text"] += (n,)
                fig_clust = go.Figure(data=[edge_trace, node_trace])
                fig_clust.update_layout(
                    height=350, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    template="plotly_white",
                )
                st.plotly_chart(fig_clust, use_container_width=True)

        # All intervention deltas table
        with st.expander("All intervention results", expanded=False):
            view = cluster_narrative_interventions.copy()
            view["key"] = view["target_cluster"] + " → " + view["affected_cluster"]
            view["max_delta"] = view[[f"delta_{k}" for k in vd_keys]].abs().max(axis=1)
            st.dataframe(view[["key", "max_delta"] + [f"delta_{k}" for k in vd_keys]].head(30), width='stretch', hide_index=True)

    else:
        st.info(
            "Cluster narrative simulation requires the quote clustering pipeline (scripts 15–17) "
            "and cluster-to-claim linking (script 23), which are only available for the synthetic "
            "dataset (`173_synthetic`). Switch to that platform in the sidebar to see opinion dynamics.\n\n"
            "```\npython src/analysis/01_run_all_platform.py\n```"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 13 — Structural Change
# ═══════════════════════════════════════════════════════════════════════════════
with tab_structural:
    if not structural_change:
        st.info("Structural change analysis not available. Run `20_structural_change_possibility.py` first.")
    else:
        scores = structural_change.get("change_readiness_scores", {})
        narrative_lines = structural_change.get("narrative", [])
        graph_summ = structural_change.get("graph_summary", {})

        st.subheader("Is structural change possible?")
        st.markdown(
            "This tab asks: *given the current relational structure, what kinds of change are feasible?* "
            "It bridges network metrics with political science frameworks to identify leverage points, "
            "blockages, path dependency, and actionable intervention strategies."
        )

        st.markdown("### Change Readiness Scores")
        st.caption("0 = rigid/locked-in, 1 = open to change. Each score has a political interpretation.")
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        score_labels = [
            ("Leverage", "leverage_score", "Are there bridges to amplify change?"),
            ("Plasticity", "plasticity_score", "Can the network accept new links?"),
            ("Blockage", "blockage_score", "How many structural barriers exist?"),
            ("Lock-in", "lockin_score", "Is a dense core blocking reconfiguration?"),
            ("Overall Readiness", "overall_readiness", "Composite of all four dimensions."),
        ]
        cols = [col_a, col_b, col_c, col_d, col_e]
        for col, (label, key, help_text) in zip(cols, score_labels):
            val = scores.get(key, 0.0)
            with col:
                st.metric(label, f"{val:.2f}", help=help_text)
                pol_narrative = scf.framework_narrative(key, val)
                if pol_narrative:
                    st.caption(pol_narrative)
                gov_action = scf.governance_action(key, val)
                if gov_action:
                    st.caption(gov_action)

        if narrative_lines:
            st.markdown("### What the scores mean for this network")
            for line in narrative_lines:
                st.write(f"- {line}")

        if graph_summ:
            st.markdown("### Network context")
            ctx_a, ctx_b, ctx_c, ctx_d = st.columns(4)
            ctx_a.metric("Nodes", graph_summ.get("nodes", 0))
            ctx_b.metric("Edges", graph_summ.get("edges", 0))
            ctx_c.metric("Components", graph_summ.get("components", 0))
            ctx_d.metric("Giant component share", f"{graph_summ.get('giant_component_share', 0):.0%}")
            cnt_e, cnt_f = st.columns(2)
            cnt_e.metric("Max k-core level", graph_summ.get("max_k_core", 0))
            cnt_f.metric("Fraction in densest core", f"{graph_summ.get('fraction_in_dense_core', 0):.0%}")

        leverage_pts = structural_change.get("leverage_points", {})
        if leverage_pts.get("top_betweenness_nodes"):
            st.markdown("### Top leverage points")
            st.caption("Nodes with highest betweenness centrality — the policy brokers (Gould & Fernandez, 1989).")
            lev_df = pd.DataFrame(leverage_pts["top_betweenness_nodes"])
            if not lev_df.empty and "global_id" in lev_df.columns:
                if "label" in nodes.columns:
                    label_map = dict(zip(nodes["global_id"].astype(str), nodes["label"]))
                    lev_df["raw_label"] = lev_df["global_id"].map(label_map)
                    lev_df["label"] = lev_df.apply(
                        lambda r: fallback_label(r["raw_label"], r["global_id"]), axis=1
                    )
                st.dataframe(
                    lev_df[["label", "betweenness_centrality", "degree"]]
                    .rename(columns={"label": "Entity", "betweenness_centrality": "Bridge score", "degree": "Links"}),
                    width='stretch', hide_index=True,
                )

        with st.expander("Metric-to-Political Mapping — reference table"):
            st.markdown(
                "How each network metric maps to a political science concept. "
                "Sources: Sabatier & Jenkins-Smith (1993), Baumgartner & Jones (1993), "
                "Gould & Fernandez (1989), Cairney (2019), Provan & Kenis (2007)."
            )
            map_df = pd.DataFrame(scf.METRIC_MAP)
            st.dataframe(
                map_df[["metric", "concept", "interpretation", "relevance"]]
                .rename(columns={"metric": "Network Metric", "concept": "Political Concept",
                                 "interpretation": "What It Means", "relevance": "Why It Matters"}),
                width='stretch', hide_index=True,
            )

        with st.expander("Governance Risk & Intervention Matrix"):
            st.markdown(
                "Translate graph findings into governance actions. "
                "Sources: Baumgartner & Jones (1993), Provan & Kenis (2007), Cairney (2019)."
            )
            gov_df = pd.DataFrame(scf.GOVERNANCE_MATRIX)
            st.dataframe(
                gov_df[["finding", "interpretation", "risk", "action"]]
                .rename(columns={"finding": "Network Finding", "interpretation": "Political Interpretation",
                                 "risk": "Governance Risk", "action": "Intervention Strategy"}),
                width='stretch', hide_index=True,
            )

        with st.expander("Intervention Strategy by Subsystem Maturity"):
            st.markdown(
                "The deep research recommends tailoring strategy to subsystem maturity:"
            )
            mat_df = pd.DataFrame(scf.INTERVENTION_MATURITY)
            st.dataframe(
                mat_df[["subsystem_type", "profile", "strategy", "alc_role"]]
                .rename(columns={"subsystem_type": "Subsystem Type", "profile": "Profile",
                                 "strategy": "Recommended Strategy", "alc_role": "ALC's Role"}),
                width='stretch', hide_index=True,
            )

        with st.expander("The three political science frameworks behind this analysis"):
            for fw in scf.FRAMEWORKS:
                st.markdown(f"**{fw['name']}** — {fw['author']}")
                st.markdown(f"*{fw['core_idea']}*")
                st.markdown(f"**Dashboard use:** {fw['dashboard_use']}")
                st.markdown(f"**Key insight:** {fw['key_insight']}")
                if "belief_hierarchy" in fw:
                    st.markdown("Belief hierarchy:")
                    for level, desc in fw["belief_hierarchy"].items():
                        st.markdown(f"- **{level}:** {desc}")
                if "pillars" in fw:
                    for name, desc in fw["pillars"].items():
                        st.markdown(f"- **{name}:** {desc}")
                if "governance_modes" in fw:
                    for mode, desc in fw["governance_modes"].items():
                        st.markdown(f"- **{mode}:** {desc}")
                st.divider()

        if not change_nodes.empty:
            with st.expander("Top nodes by k-core level"):
                st.markdown("Nodes in the highest k-core level represent the policy monopoly (Sabatier & Jenkins-Smith, 1993).")
                st.dataframe(
                    change_nodes[["global_id", "node_type", "k_core", "is_peripheral"]]
                    .rename(columns={"global_id": "Entity ID", "node_type": "Type",
                                     "k_core": "Core level", "is_peripheral": "Peripheral?"}),
                    width='stretch', hide_index=True,
                )

        with st.expander("Full research output & recommended reading"):
            st.markdown(
                "This analysis is based on the deep research report "
                "`deep_research_political_framework.md`. It synthesizes Advocacy Coalition "
                "Framework (ACF), Punctuated Equilibrium Theory (PET), and New Public "
                "Governance (NPG) to bridge network metrics to governance action."
            )
            st.markdown("**Recommended reading:**")
            st.markdown("- Sabatier & Jenkins-Smith (1993) *Policy Change and Learning*")
            st.markdown("- Baumgartner & Jones (1993) *Agendas and Instability in American Politics*")
            st.markdown("- Gould & Fernandez (1989) *Sovereignty, Conflict, and Alliance*")
            st.markdown("- Cairney (2019) *Understanding Public Policy*")
            st.markdown("- Provan & Kenis (2007) *Modes of Network Governance*")
            st.markdown("- Fischer & Maggetti (2020) *QCA and the Study of Policy Processes*")

        # ── AI-generated hypotheses ──────────────────────────────────────────
        if structural_hypotheses:
            hyps = structural_hypotheses.get("hypotheses", [])
            recs = structural_hypotheses.get("recommendations", [])
            if hyps:
                st.divider()
                st.subheader("Generated Hypotheses — What to Do")
                st.caption(
                    "Each hypothesis connects a structural metric to a specific node or claim, "
                    "grounded in the political science framework above."
                )

                # Group by type
                type_map = {
                    "leverage_brokerage": "🔗 Leverage — Brokerage Nodes",
                    "leverage_broker": "🔗 Leverage — Bridge Agents",
                    "blockage_fragile": "⚠️ Blockage — Fragile Connectors",
                    "blockage_perception": "⚠️ Blockage — Siloed Perceptions",
                    "lockin_policy_monopoly": "🔒 Lock-in — Policy Monopoly",
                    "plasticity_capacity": "🔄 Plasticity — Rewiring Capacity",
                    "plasticity_link_addition": "🔄 Plasticity — Suggested Links",
                    "narrative_cluster": "📖 Narrative Clusters",
                    "financial_perception_gap": "💰 Financial-Perception Gap",
                }
                seen_types = set()
                for h in hyps:
                    t = h.get("type", "")
                    if t not in seen_types:
                        seen_types.add(t)
                        label = type_map.get(t, t.replace("_", " ").title())
                        with st.expander(f"{label} ({sum(1 for x in hyps if x['type'] == t)} hypotheses)", expanded=(t in ("lockin_policy_monopoly", "blockage_fragile"))):
                            for sibling in hyps:
                                if sibling["type"] != t:
                                    continue
                                conf = sibling.get("confidence", "medium")
                                emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
                                with st.container(border=True):
                                    st.markdown(f"**{sibling.get('label', '')}** {emoji}")
                                    st.caption(f"Node: `{sibling.get('node_id', '')}` · Type: `{sibling.get('node_type', '')}`")
                                    st.markdown(f"*{sibling.get('hypothesis', '')}*")
                                    st.markdown(f"**Action:** {sibling.get('action', '')}")
                                    if sibling.get("metric"):
                                        st.code(sibling["metric"], language="text")

                # Ranked recommendations
                if recs:
                    st.subheader("Priority Actions")
                    st.caption("Ranked by impact and confidence.")
                    rec_df = pd.DataFrame(recs)
                    st.dataframe(
                        rec_df[["title", "action", "priority", "framework"]]
                        .rename(columns={"title": "Recommendation", "action": "Action", "priority": "Priority", "framework": "Framework"}),
                        width='stretch', hide_index=True,
                    )

        # ── Financial-perception bridge ──────────────────────────────────────
        if not financial_bridge.empty and "value_dimension" in financial_bridge.columns:
            st.divider()
            st.subheader("💶 Financial-Perception Bridge")
            st.caption(
                "Cross-tabulation of claim value dimensions with linked entities' investment data. "
                "Shows which narrative clusters are funded, under-funded, or structurally excluded."
            )

            fig_fb = px.bar(
                financial_bridge,
                x="value_dimension",
                y="total_investment_eur",
                color="mean_financial_bias",
                color_continuous_scale="RdYlGn",
                template="plotly_white",
                title="Total investment by value dimension",
                labels={"value_dimension": "Value dimension", "total_investment_eur": "Total investment (€)", "mean_financial_bias": "Financial bias"},
                text_auto=".0s",
            )
            fig_fb.update_layout(height=400)
            st.plotly_chart(fig_fb, width='stretch')

            st.dataframe(
                financial_bridge[["value_dimension", "n_claims", "n_entities", "total_investment_eur", "mean_financial_bias"]]
                .rename(columns={
                    "value_dimension": "Value dimension", "n_claims": "Claims", "n_entities": "Entities",
                    "total_investment_eur": "Total investment (€)", "mean_financial_bias": "Avg financial bias",
                }),
                width='stretch', hide_index=True,
            )

        if not narrative_budget.empty:
            st.divider()
            st.subheader("🎭 Narrative Level × Budget")
            st.caption(
                "Surface vs implicit claims and their linked budgets. "
                "Implicit claims with no budget link suggest unstated premises operate "
                "at a level detached from financial decision-making."
            )
            fig_nb = px.bar(
                narrative_budget,
                x="value_dimension",
                y="total_budget" if "total_budget" in narrative_budget.columns else "n_entities",
                color="narrative_level",
                barmode="group",
                template="plotly_white",
                title="Budget by narrative level and value dimension",
                labels={"value_dimension": "Value dimension", "total_budget": "Total budget (€)", "narrative_level": "Level"},
            )
            fig_nb.update_layout(height=350)
            st.plotly_chart(fig_nb, width='stretch')
            st.dataframe(narrative_budget, width='stretch', hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 10 — Claims
# ═══════════════════════════════════════════════════════════════════════════════
with tab_claims:
    st.header("Narrative Claims")

    st.markdown(
        """
        Claims are extracted from narrative text via hyperbase semantic hypergraph parsing.
        Each claim is structured as a **subject→verb→object** triple with entity linking to
        operational graph nodes (agents, projects). Claims span three narrative levels:
        **surface** (explicit), **implicit** (inferred via negation/conditional/emergency cues),
        and **metanarrative** (value dimension classification across 7 political domains).
        """
    )

    claim_nodes_path = ANALYSIS_DIR / "narrative_layers" / "claim_nodes.csv"
    claim_edges_path = ANALYSIS_DIR / "narrative_layers" / "claim_edges.csv"
    meta_path = ANALYSIS_DIR / "narrative_layers" / "metanarratives.csv"
    summary_path = ANALYSIS_DIR / "narrative_layers" / "narrative_extraction_summary.json"

    claim_nodes = load_csv([claim_nodes_path])
    claim_edges = load_csv([claim_edges_path])
    meta_df = load_csv([meta_path])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Claims", len(claim_nodes) if not claim_nodes.empty else 0)
    with col2:
        surface_n = int((claim_nodes["narrative_level"] == "surface").sum()) if not claim_nodes.empty else 0
        st.metric("Surface", surface_n)
    with col3:
        implicit_n = int((claim_nodes["narrative_level"] == "implicit").sum()) if not claim_nodes.empty else 0
        st.metric("Implicit", implicit_n)
    with col4:
        linked_n = int(claim_nodes["subject_entity_id"].astype(str).str.strip().ne("").sum()) if not claim_nodes.empty else 0
        st.metric("Entity-Linked", linked_n)

    if not claim_nodes.empty:
        st.subheader("Value Dimensions")
        dim_counts = (
            claim_nodes[claim_nodes["value_dimension"].notna() & (claim_nodes["value_dimension"] != "")]
            ["value_dimension"].value_counts()
        )
        if not dim_counts.empty:
            fig = px.bar(
                x=dim_counts.index, y=dim_counts.values,
                labels={"x": "Value Dimension", "y": "Claim Count"},
                color=dim_counts.index, color_discrete_sequence=PLOTLY_PALETTE,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No value dimension classifications available.")

        st.subheader("Sample Claims")

        level_filter = st.selectbox(
            "Narrative level", ["All", "surface", "implicit"],
        )

        sample_df = claim_nodes.copy()
        if level_filter != "All":
            sample_df = sample_df[sample_df["narrative_level"] == level_filter]

        display_cols = ["global_id", "narrative_level", "verb", "subject_raw", "object_raw",
                        "value_dimension", "belief_level"]
        available = [c for c in display_cols if c in sample_df.columns]
        st.dataframe(
            sample_df[available].head(50),
            width='stretch', hide_index=True,
        )

        st.subheader("Claim Graph")
        st.markdown(
            f"**{len(claim_edges)} edges** connecting narrative sources → claims → operational entities."
        )
        if not claim_edges.empty:
            edge_type_counts = claim_edges["edge_type"].value_counts()
            fig2 = px.bar(
                x=edge_type_counts.index, y=edge_type_counts.values,
                labels={"x": "Edge Type", "y": "Count"},
                color=edge_type_counts.index, color_discrete_sequence=PLOTLY_PALETTE,
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, width='stretch')

    else:
        st.warning("No narrative extraction output found. Run `21_extract_narrative_layers.py` first.")

    # Summary sidebar data
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            with st.expander("Extraction Summary"):
                st.json(summary)
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════════════════════

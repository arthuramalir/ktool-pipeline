from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="ALC K-Tool — Ecosystem Intelligence",
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
    "declared_relational": "Declared relational [from Strapi CMS: agent↔project, project↔perception, initiative interconnections]",
    "interpretive": "AI-inferred [from edges.csv: similarity, contradiction, causality, sequence between quotes]",
    "listening": "Listening [from pipeline: channel → information → value evidence chain]",
    "qualitative_narrative": "Qualitative narrative [from platform: perception ↔ challenge, pattern ↔ perception links]",
    "quote_semantic": "Quote semantic [legacy — replaced by AI-inferred]",
}

STATUS_COLORS = {
    "Robust": "#2a9d8f",
    "Underdeveloped": "#e9c46a",
    "Low coherence": "#f4a261",
    "Single channel": "#e76f51",
    "High internal contradiction": "#e63946",
    "Low purity": "#d62728",
}

PLOTLY_PALETTE = px.colors.qualitative.Set2


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def color_for_node_type(nt: str) -> str:
    return NODE_TYPE_COLORS.get(str(nt).strip().lower(), NODE_TYPE_COLORS["unknown"])


def friendly_edge_family(family: str) -> str:
    return EDGE_FAMILY_LABELS.get(str(family).strip(), str(family).replace("_", " ").title())


def family_display_from_row(row: pd.Series) -> str:
    if "edge_family_label" in row and pd.notna(row["edge_family_label"]):
        label = str(row["edge_family_label"]).strip()
        if label:
            return label
    return friendly_edge_family(row.get("edge_family", ""))


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


def load_csv(candidates: list[Path]) -> pd.DataFrame:
    for path in candidates:
        if path.exists() and path.stat().st_size > 2:
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return pd.DataFrame()


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


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.metric(label=label, value=value, help=help_text if help_text else None)


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
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

with st.sidebar:
    st.markdown("## Settings")
    platform_id = st.text_input("Platform ID", value="173").strip() or "173"
    output_subdir = st.text_input("Output subfolder", value="test").strip() or "test"
    st.caption(f"`data/processed/{platform_id}/{output_subdir}`")
    st.divider()
    is_synthetic = "synthetic" in platform_id.lower()
    if is_synthetic:
        st.success("Synthetic dataset detected — Financial Portfolio tab active")
    else:
        st.info("Financial Portfolio tab available on `173_synthetic`")

DATA_DIR = ROOT / "data" / "processed" / platform_id / output_subdir
ANALYTICS_DIR = DATA_DIR / "analytics"
ANALYSIS_DIR = DATA_DIR / "analysis"

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────
nodes = load_csv([ANALYTICS_DIR / "nodes.csv", DATA_DIR / "nodes.csv"])
edges = load_csv([ANALYTICS_DIR / "edges.csv", DATA_DIR / "edges.csv"])
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
gnn_summary = load_json(ANALYSIS_DIR / "gnn" / "gnn_summary.json")
gnn_training_report = load_json(ANALYSIS_DIR / "gnn" / "gnn_training_report.json")
gnn_link_report = load_json(ANALYSIS_DIR / "gnn" / "gnn_link_prediction_report.json")
gnn_link_recommendations = load_csv([ANALYSIS_DIR / "gnn" / "gnn_link_recommendations.csv"])
gnn_perception_effects = load_csv([ANALYSIS_DIR / "gnn" / "gnn_perception_effects.csv"])
narrative_profiles = load_csv([ANALYSIS_DIR / "narrative_profiles.csv"])
quote_clusters = load_csv([ANALYSIS_DIR / "quote_clusters.csv"])
manual_profiles = load_json(ROOT / "RESSOURCES" / "manual_narrative_profiles.json")
link_intervention_scores = load_csv([ANALYSIS_DIR / "link_intervention_scores.csv"])
link_intervention_summary = load_json(ANALYSIS_DIR / "link_intervention_summary.json")

# Financial (synthetic only)
leverage_df = load_csv([ANALYSIS_DIR / "synthetic_value_leverage.csv"])
stranded_df = load_csv([ANALYSIS_DIR / "synthetic_stranded_assets.csv"])
fin_diffusion_df = load_csv([ANALYSIS_DIR / "synthetic_financial_diffusion.csv"])
fin_summary = load_json(ANALYSIS_DIR / "synthetic_financial_summary.json")

if nodes.empty or edges.empty:
    st.error("Nodes/edges not found. Run the graph pipeline first for this platform.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("ALC K-Tool — Ecosystem Intelligence")
st.caption(
    f"Platform **{platform_id}** · subfolder **{output_subdir}** · "
    f"{len(nodes):,} entities · {len(edges):,} connections"
)

# ─────────────────────────────────────────────────────────────────────────────
# TOPLINE METRICS
# ─────────────────────────────────────────────────────────────────────────────
scope = readiness.get("graph_scope_summary", {})
topology = structural.get("topology_metrics", {})
failed_checks = int(quality_gate.get("failed_checks", 0)) if quality_gate else 0
confidence_text, confidence_level = confidence_label(failed_checks)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Entities", f"{int(scope.get('node_count', len(nodes.index))):,}")
c2.metric("Connections", f"{int(scope.get('edge_count', len(edges.index))):,}")
c3.metric("Connected groups", int(scope.get("component_count", topology.get("connected_components_count", 0))))
c4.metric("Isolated entities", int(topology.get("total_isolated_nodes", 0)))
c5.metric("Data confidence", confidence_text)
if not perception_diag.empty:
    robust_count = (perception_diag["status_flag"] == "Robust").sum()
    c6.metric("Robust perceptions", f"{robust_count}/{len(perception_diag)}")

if confidence_level == "error":
    st.error("Confidence low — use for directional insight only.")
elif confidence_level == "warning":
    st.warning("Confidence medium — some relation families need better coverage.")
else:
    st.success("Data confidence high.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_labels = [
    "Overview",
    "System Resilience",
    "Listening",
    "Narrative Profiles",
    "Stakeholder Perceptions",
    "Intervention Simulator",
    "Network Structure",
    "AI Semantic Links",
]
if is_synthetic:
    tab_labels.append("Investment Intelligence")

tabs = st.tabs(tab_labels)
tab_overview = tabs[0]
tab_alerts = tabs[1]
tab_narrative = tabs[2]
tab_profiles = tabs[3]
tab_perception = tabs[4]
tab_gnn = tabs[5]
tab_layer = tabs[6]
tab_ai_semantic = tabs[7]
tab_financial = tabs[8] if is_synthetic and len(tabs) > 8 else None


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Ecosystem shape at a glance")

    left, right = st.columns([0.38, 0.62])
    with left:
        type_counts = nodes["node_type"].astype(str).value_counts().reset_index()
        type_counts.columns = ["Node type", "Count"]
        fig_types = px.bar(
            type_counts, x="Count", y="Node type", orientation="h",
            template="plotly_white", title="Entity composition",
            color="Node type", color_discrete_sequence=PLOTLY_PALETTE,
        )
        fig_types.update_layout(height=330, margin=dict(l=10, r=10, t=42, b=10), showlegend=False)
        st.plotly_chart(fig_types, use_container_width=True)

        if not centrality.empty:
            top_connectors = centrality.copy()
            if "label" in top_connectors.columns and "global_id" in top_connectors.columns:
                top_connectors["label"] = top_connectors.apply(
                    lambda row: fallback_label(row.get("label"), row.get("global_id")), axis=1
                )
            top_connectors = top_connectors.sort_values("betweenness_centrality", ascending=False).head(8)
            st.markdown("**Top bridge connectors**")
            st.dataframe(
                top_connectors[["label", "node_type", "betweenness_centrality", "degree"]]
                .rename(columns={"label": "Entity", "node_type": "Type",
                                 "betweenness_centrality": "Bridge score", "degree": "Links"}),
                use_container_width=True, hide_index=True,
            )

    with right:
        max_nodes = st.slider("Map detail (max entities shown)", 40, 220, 120, 10)
        st.plotly_chart(network_figure(nodes, edges, max_nodes=max_nodes), use_container_width=True)

        # Node type legend
        counts = nodes["node_type"].fillna("unknown").astype(str).str.lower().value_counts().to_dict()
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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: SYSTEM RESILIENCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_alerts:
    st.subheader("Ecosystem Stress Test & Alerts")

    targeted_drop = to_float(robustness.get("systemic_drop_delta_targeted", 0.0))
    random_drop = to_float(robustness.get("systemic_drop_delta_random", 0.0))
    verdict = robustness.get("verdict", "No robustness verdict found for this run.")

    left, right = st.columns([0.5, 0.5])
    with left:
        st.markdown("**Stress test results**")
        m1, m2 = st.columns(2)
        m1.metric("Targeted attack drop", f"{targeted_drop:.1%}",
                  help="How much the main connected component shrinks when the most central nodes are removed.")
        m2.metric("Random failure drop", f"{random_drop:.1%}",
                  help="How much the main component shrinks under random node removal.")
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
                title="Core resilience under disruption",
                labels={"value": "Remaining core share", "nodes_removed_count": "Entities removed"},
            )
            fig_decay.update_layout(height=320, legend_title="Scenario")
            st.plotly_chart(fig_decay, use_container_width=True)

    with right:
        st.markdown("**Priority connectors to monitor**")
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
                use_container_width=True, hide_index=True,
            )

    st.markdown("**Recommended actions**")
    action_lines = []
    if targeted_drop > random_drop + 0.2:
        action_lines.append("Prioritize bridge redundancy: support at least 3 backup connectors for high-fragility entities.")
    orphan_project_rate = readiness.get("orphan_node_distributions", {}).get("project", {}).get("orphan_rate", 0.0)
    if to_float(orphan_project_rate) > 0.5:
        action_lines.append("Run a project-link validation sprint: many projects are disconnected from the active ecosystem.")
    if failed_checks > 0:
        action_lines.append("Improve relation capture quality for missing link families before high-stakes decisions.")
    if not action_lines:
        action_lines.append("Maintain current monitoring cadence and re-run analytics after each data refresh.")
    for idx, line in enumerate(action_lines, start=1):
        st.write(f"{idx}. {line}")

    with st.expander("How stress metrics are calculated"):
        st.markdown(
            "- **Targeted attack drop** = `1 - (largest_component_size_after / N)` after removing top-10% highest-betweenness nodes. Quantifies vulnerability to hub loss.\n"
            "- **Random failure drop** = same formula but removing 10% of nodes uniformly at random. Baseline for comparison.\n"
            "- **Vulnerable = targeted - random > 0.2** → network is significantly more fragile under targeted attack than random failure.\n"
            "- **Articulation points** = nodes whose removal increases the number of connected components.\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: LISTENING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_narrative:
    st.subheader("Narrative and listening evidence")

    c1, c2 = st.columns([0.48, 0.52])
    with c1:
        if not listening.empty:
            listening = listening.copy()
            listening["channel_display"] = listening.apply(display_channel_label, axis=1)
            channel_counts = listening["channel_display"].fillna("Unknown").astype(str).value_counts().reset_index()
            channel_counts.columns = ["Channel", "Narratives"]
            fig_ch = px.bar(
                channel_counts.head(12), x="Narratives", y="Channel", orientation="h",
                template="plotly_white", title="Narratives by channel",
            )
            fig_ch.update_layout(height=340, margin=dict(l=10, r=10, t=42, b=10))
            st.plotly_chart(fig_ch, use_container_width=True)
        else:
            st.info("Listening pipeline data not available.")

        if not diffusion.empty and "diffusion_bias" in diffusion.columns:
            diffusion_plot = diffusion.copy()
            if "label" in diffusion_plot.columns and "global_id" in diffusion_plot.columns:
                diffusion_plot["label"] = diffusion_plot.apply(
                    lambda row: fallback_label(row.get("label"), row.get("global_id")), axis=1
                )
            top_diff = diffusion_plot.sort_values("diffusion_bias", ascending=False).head(12)
            fig_diff = px.bar(
                top_diff, x="diffusion_bias", y="label", orientation="h",
                template="plotly_white", title="Narrative diffusion influence (top entities)",
            )
            fig_diff.update_layout(height=340, margin=dict(l=10, r=10, t=42, b=10), yaxis_title="")
            st.plotly_chart(fig_diff, use_container_width=True)

    with c2:
        st.markdown("**Quote explorer**")
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
                use_container_width=True, hide_index=True,
            )

    with st.expander("How narrative diffusion is measured"):
        st.markdown(
            "- **PageRank**: `PR(u) = (1-d)/N + d Σ PR(v) / out_deg(v)` — recursive importance score.\n"
            "- **Personalized PR**: replace `(1-d)/N` with `(1-d) · p(u)` where `p` is a seed-node preference vector.\n"
            "- **Diffusion bias** = `personalized_PR(u) - uniform_PR(u)`. Positive = node is preferentially reached from the seed set.\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: NARRATIVE PROFILES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_profiles:
    st.subheader("Narrative profiles from the listening layer")

    st.markdown(
        "Each **narrative profile** is a cluster of semantically related quotes extracted from the listening pipeline. "
        "The auto-generated clusters are produced by `17_cluster_quotes_into_profiles.py`: it builds a graph of quote-to-quote semantic edges "
        "(similarity, contradiction, causality from `16_detect_quote_semantic_edges.py`), computes TF-IDF features on quote text, "
        "then runs agglomerative clustering to group related statements. "
        "The **surface / implicit / meta-narrative** labels below come from `manual_narrative_profiles.json` — "
        "these are structured annotations that overlay a three-layer narrative analysis onto selected profiles."
    )

    # ── Narrative × perception topic landscape visualization
    st.markdown("### Narrative-Perception Topic Landscape")
    st.caption("Topic composition of each narrative profile, sized by quote count.")

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
                title="Topic composition by narrative profile",
                labels={"cluster": "Narrative profile", "topic": "Topic area", "count": "Quote count"},
            )
            fig_np.update_layout(height=400, showlegend=False, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_np, use_container_width=True)

            # Topic-perception overlay
            st.caption("Topic overlap: how narrative themes connect to the perception layer.")
            topic_pivot = topic_df.groupby("topic")["cluster_id"].nunique().reset_index(name="n_profiles")
            topic_pivot = topic_pivot.sort_values("n_profiles", ascending=False).head(15)
            fig_topic = px.bar(
                topic_pivot, x="n_profiles", y="topic", orientation="h",
                template="plotly_white", color="n_profiles", color_continuous_scale="Blues",
                title="Topics spanning multiple narrative profiles",
                labels={"n_profiles": "Number of profiles", "topic": "Topic"},
            )
            fig_topic.update_layout(height=320, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_topic, use_container_width=True)
    else:
        st.info("Run the quote clustering pipeline first.")

    st.divider()

    if isinstance(manual_profiles, list) and manual_profiles:
        for prof in manual_profiles:
            with st.expander(f"**{prof.get('name', 'Unnamed')}** — channels: {', '.join(prof.get('source_channels', []))}"):
                c1, c2 = st.columns([0.5, 0.5])
                with c1:
                    st.markdown(f"**Key idea** — {prof.get('key_idea', '')}")
                    st.markdown(f"**Surface narrative:** {prof.get('surface_narrative', '')}")
                    st.markdown(f"**Implicit narrative:** {prof.get('implicit_narrative', '')}")
                    st.markdown(f"**Meta-narrative:** {prof.get('metanarrative', '')}")
                with c2:
                    st.markdown("**Representative quotes:**")
                    for q in prof.get("representative_quotes", []):
                        st.markdown(f"- _{q}_")
                    st.markdown(f"**Emotional tone:** {prof.get('emotional_tone', '')}")
                    st.markdown(f"**Associated values:** {', '.join(prof.get('associated_values', []))}")
                    if prof.get("contradiction_with"):
                        st.markdown(f"**Contradicts:** {', '.join(prof['contradiction_with'])}")
    else:
        st.info("No manual narrative profiles found. Add profiles to `RESSOURCES/manual_narrative_profiles.json`.")

    st.divider()

    # ── Surface / Implicit / Meta-narrative illustration
    with st.expander("📊 Three-layer narrative analysis — quick reference guide"):
        st.markdown(
            "Each narrative profile can be annotated with three interpretative layers. "
            "This framework helps move from *what people say* to *what the narrative reveals about power, values, and structure*."
        )
        st.markdown("---")

        col_s, col_i, col_m = st.columns(3)
        with col_s:
            st.markdown("**🔹 Surface narrative**")
            st.markdown("*What is explicitly stated*")
            st.success(
                "The literal claim or description. "
                "Usually summarises the direct quote content without inference."
            )
            st.markdown("How to identify:")
            st.markdown("- Paraphrase the quote(s) in one sentence\n- Stay close to the text\n- Avoid reading between the lines")
            st.markdown("---")
            st.info("_Example:_\n'Funding is short-term and creates uncertainty for planning long-term projects.'")

        with col_i:
            st.markdown("**🔸 Implicit narrative**")
            st.markdown("*What is assumed or implied*")
            st.warning(
                "The belief, assumption, or causal logic hiding underneath. "
                "Answers: 'what must be true for this statement to make sense?'"
            )
            st.markdown("How to identify:")
            st.markdown("- Ask: 'what does this take for granted?'\n- Look for unstated causation\n- Surface the value judgement")
            st.markdown("---")
            st.info("_Example:_\n'Short-term funding implies funders don't trust communities to manage multi-year budgets.'")

        with col_m:
            st.markdown("**🔶 Meta-narrative**")
            st.markdown("*The deeper cultural/power frame*")
            st.error(
                "The systemic logic or worldview that structures the narrative. "
                "Answers: 'what power dynamic or institutional logic does this reinforce or resist?'"
            )
            st.markdown("How to identify:")
            st.markdown("- Connect to broader social/political context\n- Identify whose interests are served\n- Surface the structural contradiction")
            st.markdown("---")
            st.info("_Example:_\n'Performance-based funding regimes produce precarity under the appearance of accountability — a governance technology that masks withdrawal of core support.'")

        st.divider()
        st.markdown(
            "**How to use this in profile creation:**\n\n"
            "1. Start with the **surface** — pull the direct claim from the representative quote.\n"
            "2. Ask **'what is taken for granted?'** to surface the **implicit** layer.\n"
            "3. Ask **'what systemic logic does this connect to?'** to reach the **meta-narrative**.\n"
            "4. The three layers should form a coherent stack: surface → implicit → meta (each deeper layer explains why the layer above exists)."
        )

    st.divider()
    st.markdown("### Auto-generated profile clusters")

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
            st.dataframe(view, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: STAKEHOLDER PERCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_perception:
    st.subheader("Perception Diagnostics")
    st.caption(
        "Quantitative health metrics for each perception archetype in the K-Tool Listening module. "
        "Based on LLM-generated semantic edge weights between citizen quotes."
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
        m2.metric("Robust", f"{n_robust}/{n_perceptions}")
        m3.metric("Avg coherence", f"{avg_coherence:.2f}" if avg_coherence is not None else "n/a",
                  help="Mean similarity of LLM edge weights within each perception. >0.6 = strong internal agreement.")
        m4.metric("Avg purity", f"{avg_purity:.2f}" if avg_purity is not None else "n/a",
                  help="Fraction of a quote's top-5 semantic neighbours that belong to the same perception.")

        st.divider()

        # ── Colour-coded status table
        st.markdown("**Perception health table** (sorted by quote count)")

        def _status_colour(flag: str) -> str:
            if flag == "Robust":
                return "background-color: #d4edda; color: #155724;"
            if "Underdeveloped" in flag:
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
            "internal_coherence": "Coherence ↑",
            "purity_score": "Purity ↑",
            "source_entropy": "Source entropy",
            "contradiction_density": "Contradiction %",
            "status_flag": "Status",
        }
        diag_view = diag_view[[c for c in display_rename if c in diag_view.columns]].rename(columns=display_rename)

        # Format floats
        for col in ["Coherence ↑", "Purity ↑", "Source entropy", "Contradiction %"]:
            if col in diag_view.columns:
                diag_view[col] = diag_view[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

        # Style status column
        def _style_row(row):
            flag = row.get("Status", "")
            style = _status_colour(str(flag))
            return [style if col == "Status" else "" for col in row.index]

        styled = diag_view.style.apply(_style_row, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

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
                title="Perception Coherence vs Purity",
                labels={"internal_coherence": "Internal Coherence (LLM edge weights)",
                        "purity_score": "Purity Score (% top neighbours in same perception)"},
                color_discrete_sequence=PLOTLY_PALETTE,
            )
            fig_scatter.add_vline(x=0.4, line_dash="dot", line_color="orange",
                                  annotation_text="Coherence threshold", annotation_position="top left")
            fig_scatter.add_hline(y=0.3, line_dash="dot", line_color="orange",
                                  annotation_text="Purity threshold", annotation_position="bottom right")
            fig_scatter.update_traces(textposition="top center")
            fig_scatter.update_layout(height=480, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Source entropy bar chart
        if "source_entropy" in perception_diag.columns and "perception_label" in perception_diag.columns:
            ent_df = perception_diag[["perception_label", "source_entropy", "n_channels"]].dropna().copy()
            ent_df["perception_label"] = ent_df["perception_label"].astype(str).str[:50]
            fig_entropy = px.bar(
                ent_df.sort_values("source_entropy"),
                x="source_entropy", y="perception_label", orientation="h",
                template="plotly_white",
                title="Source Channel Entropy per Perception",
                labels={"source_entropy": "Shannon entropy (higher = more diverse channels)",
                        "perception_label": "Perception"},
                color="n_channels", color_continuous_scale="Blues",
            )
            fig_entropy.add_vline(x=0.5, line_dash="dot", line_color="red",
                                  annotation_text="Single-channel risk threshold")
            fig_entropy.update_layout(height=360, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_entropy, use_container_width=True)

        st.info(
            "**How to interpret these metrics:**\n\n"
            "- **Coherence < 0.4** → Quotes in this perception disagree semantically; consider splitting it.\n"
            "- **Purity < 0.3** → Quotes are closer to other perceptions than their own; possible mis-assignment.\n"
            "- **Source entropy < 0.5** → A single interview channel dominates; the perception may reflect one interviewer's framing, not a systemic narrative.\n"
            "- **Contradiction > 30%** → Strongly contested perception — valuable to surface, but harder to interpret as a unified viewpoint."
        )

        with st.expander("How each metric is calculated"):
            st.markdown(
                "- **Coherence** = mean cosine similarity of all within-perception quote embedding pairs. High (>0.6) = quotes agree on direction.\n"
                "- **Purity** = for each quote, fraction of its top-5 semantic neighbours that share the same perception label. Low (<0.3) = boundary case.\n"
                "- **Source entropy** = `H = -Σ p(c) log₂ p(c)` over channels feeding the perception. Low (<0.5) = one channel dominates.\n"
                "- **Contradiction density** = fraction of within-perception semantic edges labelled `contradiction` out of all its intra-perception edges.\n"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: INTERVENTION SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_gnn:
    st.subheader("GNN preparation summary")
    st.caption("Prepared tensors for heterogeneous graph learning and node-level training." )

    if not gnn_link_recommendations.empty:
        st.markdown("**1. Mapping AI: top 3 mapping-layer links**")
        aid_view = gnn_link_recommendations.copy()
        show_cols = [
            c for c in [
                "source_label",
                "target_label",
                "pair_kind",
                "link_type",
                "recommendation_category",
                "impact_layer",
                "link_probability",
                "rationale",
            ] if c in aid_view.columns
        ]
        if show_cols:
            st.dataframe(
                aid_view[show_cols].head(3).rename(columns={
                    "source_label": "From",
                    "target_label": "To",
                    "pair_kind": "Pair kind",
                    "link_type": "Link type",
                    "recommendation_category": "Why this helps",
                    "impact_layer": "Impact layer",
                    "link_probability": "Score",
                    "rationale": "Dashboard note",
                }),
                use_container_width=True,
                hide_index=True,
            )
        st.caption("This is the mapping layer output. It answers which agent/project links are worth considering before any narrative-space analysis.")

    if not gnn_perception_effects.empty:
        st.markdown("**2. Effect on perception spaces**")
        impact_view = gnn_perception_effects.copy()
        show_cols = [
            c for c in [
                "source_label",
                "target_label",
                "perception_effect_type",
                "source_perception_count",
                "target_perception_count",
                "shared_perception_count",
                "perception_overlap_ratio",
                "perception_effect_note",
            ] if c in impact_view.columns
        ]
        if show_cols:
            impact_view = impact_view[show_cols].rename(columns={
                "source_label": "From",
                "target_label": "To",
                "perception_effect_type": "Perception effect",
                "source_perception_count": "Source perceptions",
                "target_perception_count": "Target perceptions",
                "shared_perception_count": "Shared perceptions",
                "perception_overlap_ratio": "Overlap ratio",
                "perception_effect_note": "Effect note",
            })
            impact_view["Overlap ratio"] = impact_view["Overlap ratio"].apply(lambda value: f"{float(value):.2f}" if pd.notna(value) else "—")
            st.dataframe(impact_view.head(3), use_container_width=True, hide_index=True)
        st.caption("This is the second layer. It asks how a proposed mapping link could reinforce, bridge, or extend the perception space downstream.")

    if not link_intervention_scores.empty:
        st.divider()
        st.markdown("**3. Link intervention simulation**")
        st.caption("Simulates adding each proposed link and measures the structural impact on the perception space.")

        if link_intervention_summary:
            m1, m2, m3 = st.columns(3)
            m1.metric("Simulated links", int(link_intervention_summary.get("n_recommendations_simulated", 0)))
            m2.metric("Merge components", int(link_intervention_summary.get("n_merging_components", 0)),
                      help="How many proposed links connect two previously separate graph components.")
            m3.metric("Avg sensitivity", f"{link_intervention_summary.get('avg_sensitivity', 0.0):.3f}",
                      help="Composite score: higher = the structural change has more impact on narrative influence distribution.")

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
                use_container_width=True, hide_index=True,
            )

        with st.expander("How intervention sensitivity is calculated"):
            st.markdown(
                "- **Sensitivity** = `|avg PR delta| × 10 + 0.3 if merges components + reach_expansion / N_info + min(|max PR delta| × 50, 0.2)`\n"
                "- **PR delta** = `simulated_PageRank(u) - baseline_PageRank(u)` for each perception node. Measures narrative influence shift.\n"
                "- **Merges components** = the proposed link connects two previously disconnected parts of the graph — a structural bridge event.\n"
                "- **Info reach expansion** = number of information nodes that become reachable from any perception after adding the link (within 3 hops).\n"
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
        m4.metric("Feature dim", int(gnn_summary.get('feature_dim', 0)))

        c1, c2 = st.columns([0.58, 0.42])
        with c1:
            semantic_coverage = float(gnn_summary.get('semantic_embedding_coverage_rate', 0.0))
            semantic_source = gnn_summary.get('semantic_embedding_source') or "None found; using text-hash features"
            summary_items = {
                "Semantic coverage": f"{semantic_coverage:.1%}",
                "Low-confidence edges": int(gnn_summary.get('low_confidence_edge_count', 0)),
                "Low-confidence threshold": gnn_summary.get('low_confidence_threshold', 0.0),
                "Semantic source": semantic_source,
            }
            st.markdown("**Preparation summary**")
            st.dataframe(pd.DataFrame(list(summary_items.items()), columns=["Metric", "Value"]), use_container_width=True, hide_index=True)
            if semantic_coverage == 0.0:
                st.info("Semantic coverage is 0.0% because no external embedding file was found for this run. The GNN prep is using text-hash features and structural/categorical fields instead.")

            if gnn_summary.get("feature_slices"):
                st.markdown("**Feature slices**")
                feature_slice_rows = []
                for name, bounds in gnn_summary.get("feature_slices", {}).items():
                    feature_slice_rows.append(
                        {
                            "Block": name,
                            "Start": bounds.get("start", 0),
                            "End": bounds.get("end", 0),
                            "Width": int(bounds.get("end", 0)) - int(bounds.get("start", 0)),
                        }
                    )
                st.dataframe(pd.DataFrame(feature_slice_rows), use_container_width=True, hide_index=True)

        with c2:
            st.markdown("**Pipeline stages**")
            for stage in gnn_summary.get("pipeline_stages", []):
                st.write(f"- {stage}")
            if gnn_summary.get("privacy_notes"):
                st.markdown("**Privacy notes**")
                for note in gnn_summary.get("privacy_notes", []):
                    st.write(f"- {note}")

        if gnn_summary.get("sensitive_columns_used"):
            st.info(f"Sensitive columns projected out or encoded: {', '.join(gnn_summary.get('sensitive_columns_used', []))}")

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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: NETWORK STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_layer:
    st.subheader("What each edge family contributes")

    layer_metrics = readiness.get("layer_family_analysis", {})
    if layer_metrics:
        layer_df = pd.DataFrame([
            {
                "Edge family": friendly_edge_family(layer),
                "Connections": vals.get("edge_count", 0),
                "Connected groups": vals.get("component_count", 0),
                "Density": vals.get("graph_density", 0.0),
            }
            for layer, vals in layer_metrics.items()
        ])
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(layer_df, x="Edge family", y="Connections",
                                   template="plotly_white", title="Connections by edge family"), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(layer_df, x="Edge family", y="Connected groups",
                                   template="plotly_white", title="Fragmentation by edge family"), use_container_width=True)
    else:
        st.info("Edge family report not available for this run.")

    st.markdown("**Interactive edge family filter**")
    family_source_column = "edge_family_label" if "edge_family_label" in edges.columns else "edge_family"
    available_families_raw = sorted(edges[family_source_column].dropna().astype(str).unique().tolist()) if family_source_column in edges.columns else []
    family_display_map = {k: friendly_edge_family(k) for k in available_families_raw}
    available_family_display = list(family_display_map.values())
    selected_families = st.multiselect(
        "Choose edge families to include in the graph below",
        options=available_family_display,
        default=available_family_display,
        help="Uncheck an edge family to hide those connections and see how the remaining graph changes.",
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
    m1.metric("Filtered connections", int(filtered_edges.shape[0]))
    m2.metric("Connected groups", int(component_count))
    m3.metric("Largest group share", f"{largest_share:.1%}")
    st.caption("If adding an edge family reduces connected groups and grows the largest share, it reveals hidden links.")

    with st.expander("About edge families"):
        st.markdown(
            "An **edge family** is a group of connections that share the same origin. "
            "Each represents a different way of knowing about the ecosystem:\n\n"
            "| Edge family | What it records | How it was built | Provenance |\n"
            "| --- | --- | --- | --- |\n"
            "| **Declared relational** | Agent↔project, project↔perception, initiative interconnections | Extracted directly from Strapi CMS relation fields | Source platform |\n"
            "| **AI-inferred** | Quote-to-quote semantic links (similarity, contradiction, causality, sequence) | `14_alc_advanced_semantic_edges.py` — sentence embeddings + rule-based classification | Algorithmic inference |\n"
            "| **Listening** | Channel→information→value evidence chains | Reconstructed from extraction pipeline relational joins | Pipeline derivation |\n"
            "| **Qualitative narrative** | Perception↔challenge, pattern↔perception links | Extracted from Strapi CMS structured qualitative fields | Source platform |\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8: AI SEMANTIC LINKS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ai_semantic:
    st.subheader("AI-generated semantic links")
    st.caption("Links inferred by the LLM semantic layer — separated so non-technical users can distinguish them from declared source-data relationships.")

    ai_edges = edges.copy()
    ai_mask = pd.Series([False] * len(ai_edges), index=ai_edges.index)
    if "is_ai_generated" in ai_edges.columns:
        ai_mask = ai_edges["is_ai_generated"].apply(normalize_bool)
    if "edge_origin" in ai_edges.columns:
        ai_mask = ai_mask | ai_edges["edge_origin"].astype(str).str.lower().str.contains("ai|semantic|inference", regex=True)
    ai_edges = ai_edges[ai_mask].copy()

    if ai_edges.empty:
        st.info("No AI semantic links found. Run the semantic edge stage first.")
    else:
        ai_total = len(ai_edges)
        ai_share = ai_total / len(edges) if len(edges) else 0
        type_col = "edge_type" if "edge_type" in ai_edges.columns else None

        m1, m2, m3 = st.columns(3)
        m1.metric("AI semantic links", ai_total)
        m2.metric("Share of all links", f"{ai_share:.1%}")
        if type_col:
            m3.metric("Semantic types", int(ai_edges[type_col].nunique()))

        c1, c2 = st.columns(2)
        with c1:
            if type_col:
                tc = ai_edges[type_col].fillna("unknown").astype(str).value_counts().reset_index()
                tc.columns = ["Type", "Count"]
                fig_t = px.bar(tc, x="Count", y="Type", orientation="h", template="plotly_white",
                               title="AI links by semantic type", color="Type",
                               color_discrete_sequence=PLOTLY_PALETTE)
                fig_t.update_layout(height=360, showlegend=False)
                st.plotly_chart(fig_t, use_container_width=True)
        with c2:
            param_col = next((c for c in ["alc_semantic_parameter", "parameter"] if c in ai_edges.columns), None)
            if param_col:
                pc = ai_edges[param_col].fillna("unknown").astype(str).value_counts().reset_index()
                pc.columns = ["Parameter", "Count"]
                fig_p = px.bar(pc.head(12), x="Count", y="Parameter", orientation="h",
                               template="plotly_white", title="AI semantic parameters")
                fig_p.update_layout(height=360)
                st.plotly_chart(fig_p, use_container_width=True)

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
                use_container_width=True, hide_index=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9: INVESTMENT INTELLIGENCE (173_synthetic only)
# ═══════════════════════════════════════════════════════════════════════════════
if tab_financial is not None:
    with tab_financial:
        st.subheader("Financial Portfolio Analysis")
        st.caption(
            "Synthetic dataset: same network topology as real Platform 173 but with "
            "deterministic budget (initiatives) and investment estimates (agents) added. "
            "Run `04_investment_opportunity_synthetic.py` to generate data."
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
                m1.metric("Initiatives with budget", fin_summary.get("n_initiatives_with_budget", "—"))
                m2.metric("Agents with investment est.", fin_summary.get("n_agents_with_investment", "—"))
                m3.metric("Stranded assets", fin_summary.get("stranded_assets_count", "—"),
                          help="Initiatives with budget ≥ €500k but zero bridge score and ≤ 3 connections.")
                if not leverage_df.empty and "leverage_score" in leverage_df.columns:
                    top_leverage = leverage_df["leverage_score"].max()
                    m4.metric("Max leverage score", f"{top_leverage:.3f}",
                              help="Betweenness centrality per €1M budget — higher = more impact per euro.")

            st.divider()

            if not leverage_df.empty:
                st.markdown("### A. Value Leverage — Budget vs Bridge Score")
                st.caption("High-leverage initiatives are structurally central (bridge many disconnected groups) but relatively cheap.")
                if "associated_budget" in leverage_df.columns and "betweenness_centrality" in leverage_df.columns:
                    lev_plot = leverage_df.copy()
                    lev_plot["label"] = lev_plot.get("label", lev_plot["global_id"]).astype(str).str[:35]
                    lev_plot["budget_M"] = lev_plot["associated_budget"] / 1_000_000
                    color_col = "investment_level" if "investment_level" in lev_plot.columns else "node_type"
                    fig_lev = px.scatter(
                        lev_plot, x="budget_M", y="betweenness_centrality",
                        color=color_col, size="leverage_score", hover_name="label",
                        template="plotly_white",
                        title="Budget (€M) vs Network Bridge Score",
                        labels={"budget_M": "Budget (€M)", "betweenness_centrality": "Bridge score (betweenness)"},
                        color_discrete_sequence=PLOTLY_PALETTE,
                    )
                    fig_lev.update_layout(height=480, legend_title=color_col.replace("_", " ").title())
                    st.plotly_chart(fig_lev, use_container_width=True)
                    st.markdown("**Top 10 high-leverage initiatives**")
                    show_cols = [c for c in ["label", "node_type", "investment_level", "associated_budget",
                                             "betweenness_centrality", "leverage_score"] if c in leverage_df.columns]
                    st.dataframe(
                        leverage_df[show_cols].head(10).rename(columns={
                            "label": "Initiative", "node_type": "Type",
                            "investment_level": "Budget tier", "associated_budget": "Budget (€)",
                            "betweenness_centrality": "Bridge score", "leverage_score": "Leverage",
                        }), use_container_width=True, hide_index=True,
                    )

            st.divider()

            if not stranded_df.empty:
                st.markdown("### B. Stranded Assets")
                st.caption("High-budget initiatives with minimal network integration (bridge score ≈ 0, degree ≤ 3). They represent financial investment not connected to ecosystem flows.")
                st.error(f"{len(stranded_df)} stranded assets detected")
                show_cols = [c for c in ["label", "node_type", "associated_budget", "betweenness_centrality",
                                         "degree", "stranded_reason"] if c in stranded_df.columns]
                st.dataframe(stranded_df[show_cols].rename(columns={
                    "label": "Initiative", "node_type": "Type", "associated_budget": "Budget (€)",
                    "betweenness_centrality": "Bridge score", "degree": "Connections", "stranded_reason": "Flag",
                }), use_container_width=True, hide_index=True)
            else:
                st.success("No stranded assets detected.")

            st.divider()

            if not fin_diffusion_df.empty and "financial_bias" in fin_diffusion_df.columns:
                st.markdown("### C. Financial-Weighted Narrative Diffusion")
                st.caption(
                    "Personalized PageRank seeded by agent investment amounts (€). "
                    "Positive bias = this node is preferentially reached by financial capital flows. "
                    "Negative bias = financially under-reached, despite being part of the network."
                )
                fig_fd = px.histogram(
                    fin_diffusion_df, x="financial_bias", color="node_type",
                    nbins=30, barmode="overlay", opacity=0.6,
                    template="plotly_white",
                    title="Financial Bias Distribution by Node Type",
                    labels={"financial_bias": "Financial bias (+ = capital-heavy narrative zone)"},
                    color_discrete_sequence=PLOTLY_PALETTE,
                )
                fig_fd.add_vline(x=0, line_dash="dash", line_color="grey")
                fig_fd.update_layout(height=380, bargap=0.05)
                st.plotly_chart(fig_fd, use_container_width=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Over-reached by financial capital (bias > 0.3)**")
                    top_fin = fin_diffusion_df[fin_diffusion_df["financial_bias"] > 0.3].nlargest(8, "financial_bias")
                    if not top_fin.empty:
                        show = [c for c in ["label", "node_type", "financial_bias"] if c in top_fin.columns]
                        st.dataframe(top_fin[show].rename(columns={"label": "Entity", "node_type": "Type", "financial_bias": "Bias"}),
                                     use_container_width=True, hide_index=True)
                    else:
                        st.info("No strongly over-reached nodes.")
                with col_b:
                    st.markdown("**Under-reached by financial capital (bias < −0.3)**")
                    bot_fin = fin_diffusion_df[fin_diffusion_df["financial_bias"] < -0.3].nsmallest(8, "financial_bias")
                    if not bot_fin.empty:
                        show = [c for c in ["label", "node_type", "financial_bias"] if c in bot_fin.columns]
                        st.dataframe(bot_fin[show].rename(columns={"label": "Entity", "node_type": "Type", "financial_bias": "Bias"}),
                                     use_container_width=True, hide_index=True)
                    else:
                        st.info("No strongly under-reached nodes.")

            st.divider()

            st.markdown("### D. Budget Reallocation Simulation")
            st.caption("Shuffles budgets across initiatives 1,000 times to test whether the real allocation is better than random at placing money on structurally central nodes.")
            realloc = fin_summary.get("budget_reallocation", {}) if fin_summary else {}
            if realloc:
                c1, c2, c3 = st.columns(3)
                c1.metric("Observed mean leverage", f"{realloc.get('observed_mean_leverage', 0):.5f}",
                          delta=f"p={realloc.get('mean_leverage_percentile', 0):.0f}%", delta_color="inverse",
                          help="Lower percentile = budget is on less central nodes than random expectation.")
                c2.metric("Observed Gini", f"{realloc.get('observed_gini', 0):.3f}",
                          delta=f"p={realloc.get('gini_percentile', 0):.0f}%", delta_color="inverse",
                          help="Gini coefficient of leverage — higher = more unequal distribution than random.")
                c3.metric("Stranded assets", f"{realloc.get('observed_stranded', 0)}",
                          delta=f"p={realloc.get('stranded_percentile', 0):.0f}%", delta_color="inverse",
                          help=f"Sim mean: {realloc.get('sim_mean_stranded', 0):.1f}")
                st.error(
                    f"**Verdict: {realloc.get('efficiency_verdict', 'N/A').replace('_', ' ').title()}** — "
                    f"the observed mean leverage sits at the **{realloc.get('mean_leverage_percentile', 0):.0f}th percentile** "
                    f"({realloc.get('n_simulations', 0)} simulations). "
                )
                sim_plot_path = ANALYSIS_DIR / "financial_plots" / "budget_reallocation_simulation.png"
                if sim_plot_path.exists():
                    st.image(str(sim_plot_path), use_container_width=True)
                with st.expander("How the simulation works"):
                    st.markdown(
                        "- Budgets are randomly permuted across initiatives (preserving distribution, not assignment).\n"
                        "- Tracks: mean leverage, Gini coefficient, stranded count.\n"
                        "- p < 5% or p > 95% means the real allocation is significantly different from random.\n"
                    )

            st.divider()

            st.markdown("### E. Narrative Profile Budget Exposure")
            st.caption("For each narrative profile, traces through shared topics to estimate which initiatives' budgets are associated.")
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
                        template="plotly_white", title="Total associated budget per narrative profile",
                        labels={"cluster": "Narrative profile", "total_budget": "Associated budget (€)", "mean_leverage": "Mean leverage"})
                    fig_ne.update_layout(height=350)
                    st.plotly_chart(fig_ne, use_container_width=True)
                    show_cols = ["cluster", "n_initiatives", "total_budget", "mean_leverage", "mean_bridge"]
                    st.dataframe(profile_fin[show_cols].rename(columns={
                        "cluster": "Profile", "n_initiatives": "Initiatives", "total_budget": "Total budget (€)",
                        "mean_leverage": "Mean leverage", "mean_bridge": "Mean bridge"}), use_container_width=True, hide_index=True)
                    st.caption("**Note:** Uses fuzzy topic matching — indicative, not a direct graph-traversal link.")
                else:
                    st.info("No topic-matched initiatives found for narrative profiles.")
            else:
                st.info("Run the narrative profile pipeline and financial analysis first.")

            st.divider()

            if not link_intervention_scores.empty:
                st.markdown("### F. Link Intervention Sensitivity")
                st.caption("Simulates adding each GNN-proposed link. High-sensitivity = high-risk/high-impact allocation decisions.")
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
                        "avg_perception_PR_delta": "Avg PR delta"}), use_container_width=True, hide_index=True)
                with st.expander("How sensitivity is calculated"):
                    st.markdown("- Sensitivity = `|avg PR delta| × 10 + 0.3 if merges components + reach_expansion / N_info + min(|max PR delta| × 50, 0.2)`")

            st.info(
                "**Research interpretation:**\n\n"
                "The synthetic financial layer tests a core K-Tool hypothesis: *does financial investment track "
                "network centrality?* If stranded assets exist (high budget, low bridge), it suggests "
                "that resource allocation in the ecosystem is not optimised for systemic impact.\n\n"
                "The budget reallocation simulation confirms this: the observed allocation is **worse than random** "
                "(mean leverage at 1st percentile), meaning the current distribution of financial resources is "
                "statistically unlikely to maximise network bridge value.\n\n"
                "The financial-weighted diffusion reveals which narrative spaces are dominated by capital flows, "
                "operationalizing the concept of institutional narrative capture."
            )

# ═══════════════════════════════════════════════════════════════════════════════

"""Structural analysis hypotheses generator — connects political framework to real nodes.

Reads existing analysis outputs (centrality, k-core, robustness, GNN predictions,
narrative layers, financial data) and generates concrete, actionable hypotheses.

Two outputs:
  1. `structural_hypotheses.json` — named hypotheses referencing specific node IDs
  2. `financial_perception_bridge.csv` — cross-tabulation of value dimensions × budget

Usage:
    set KTOOL_PLATFORM_ID=173_synthetic & set KTOOL_OUTPUT_SUBDIR=test
    python src/analysis/22_structural_analysis_hypotheses.py
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from graph_utils import (
    ANALYSIS_DIR,
    ANALYTICS_DIR,
    build_graph,
    read_csv_safe,
    write_frame,
    write_json,
)


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 2:
        return pd.read_csv(path)
    return pd.DataFrame()


def label_for(nid: str, node_labels: dict[str, str]) -> str:
    return node_labels.get(nid, nid)


def main() -> None:
    print("Loading graph and analysis outputs...")

    G_multi = build_graph(directed=False)
    G = G_multi.to_undirected() if hasattr(G_multi, "to_undirected") else G_multi

    if G.number_of_nodes() == 0:
        print("ERROR: empty graph.")
        return

    # ── Load all data sources ────────────────────────────────────────────────
    structural_change = load_json(ANALYSIS_DIR / "structural_change_possibility.json")
    claim_nodes = load_csv(ANALYSIS_DIR / "narrative_layers" / "claim_nodes.csv")
    claim_edges = load_csv(ANALYSIS_DIR / "narrative_layers" / "claim_edges.csv")
    nodes_df, edges_df = read_csv_safe(ANALYTICS_DIR / "nodes.csv"), read_csv_safe(ANALYTICS_DIR / "edges.csv")
    centrality = load_csv(ANALYSIS_DIR / "node_centrality.csv")
    kcore_df = load_csv(ANALYSIS_DIR / "kcore_membership.csv")
    bridge_agents = load_csv(ANALYSIS_DIR / "bridge_agents.csv")
    fragility = load_csv(ANALYSIS_DIR / "top_fragility_nodes.csv")
    vulnerable = load_csv(ANALYSIS_DIR / "vulnerable_connectors.csv")
    gnn_recs = load_csv(ANALYSIS_DIR / "gnn" / "gnn_link_recommendations.csv")
    readiness_nodes = load_csv(ANALYSIS_DIR / "change_readiness_nodes.csv")

    # Financial data
    fin_diffusion = load_csv(ANALYSIS_DIR / "synthetic_financial_diffusion.csv")
    value_leverage = load_csv(ANALYSIS_DIR / "synthetic_value_leverage.csv")

    # Node lookup
    node_labels = dict(zip(nodes_df["global_id"].astype(str), nodes_df["label"])) if not nodes_df.empty else {}
    node_types = dict(zip(nodes_df["global_id"].astype(str), nodes_df["node_type"])) if not nodes_df.empty else {}

    hypotheses: list[dict] = []
    recommendations: list[dict] = []

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. LEVERAGE HYPOTHESES — boundary spanning & brokerage
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n--- Leverage hypotheses ---")

    if isinstance(structural_change, dict):
        top_betweenness = structural_change.get("leverage_points", {}).get("top_betweenness_nodes", [])
        top_brokers = structural_change.get("leverage_points", {}).get("top_bridge_agents", [])

        for node in top_betweenness:
            nid = node.get("global_id", "")
            nt = node_types.get(nid, "unknown")
            bc = node.get("betweenness_centrality", 0)
            deg = node.get("degree", 0)
            lb = label_for(nid, node_labels)
            hypotheses.append({
                "id": f"H1_{nid.replace('.', '_')}",
                "type": "leverage_brokerage",
                "node_id": nid,
                "node_type": nt,
                "label": lb,
                "metric": f"betweenness={bc:.4f}, degree={deg}",
                "hypothesis": (
                    f"{lb} ({nt}) mediates information across structural holes "
                    f"(betweenness={bc:.4f}, degree={deg}). "
                    f"Strengthening its brokerage role could accelerate narrative and resource "
                    f"diffusion across components."
                ),
                "framework": "Policy Brokerage / Boundary Spanning",
                "action": "Fund coordination capacity; protect from turnover; assign liaison role.",
                "confidence": "medium" if bc < 0.005 else "high",
            })

        for node in top_brokers:
            nid = node.get("global_id", "")
            score = node.get("score", 0)
            lb = label_for(nid, node_labels)
            nt = node_types.get(nid, "unknown")
            hypotheses.append({
                "id": f"H2_{nid.replace('.', '_')}",
                "type": "leverage_broker",
                "node_id": nid,
                "node_type": nt,
                "label": lb,
                "metric": f"bridge_score={score:.4f}",
                "hypothesis": (
                    f"{lb} ({nt}) is a top bridge agent (score={score:.4f}). "
                    f"This actor connects otherwise disconnected groups. "
                    f"Losing them would fragment coordination."
                ),
                "framework": "Policy Brokerage / Boundary Spanning",
                "action": "Integrate into steering committees; ensure knowledge transfer; designate backup.",
                "confidence": "high" if score > 0.32 else "medium",
            })

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. BLOCKAGE HYPOTHESES — fragile connectors & isolated perceptions
    # ═══════════════════════════════════════════════════════════════════════════
    print("--- Blockage hypotheses ---")

    # Fragile connectors
    if not vulnerable.empty:
        for _, row in vulnerable.head(10).iterrows():
            nid = str(row.get("global_id", ""))
            lb = label_for(nid, node_labels)
            nt = node_types.get(nid, "unknown")
            hypotheses.append({
                "id": f"H3_{nid.replace('.', '_')}",
                "type": "blockage_fragile",
                "node_id": nid,
                "node_type": nt,
                "label": lb,
                "metric": "articulation_point",
                "hypothesis": (
                    f"{lb} ({nt}) is an articulation point — its removal disconnects the network. "
                    f"If {lb} leaves, information flow between components will be severed."
                ),
                "framework": "Institutional Bottlenecks / Structural Cleavages",
                "action": "Engineer redundant connections around this node. Fund bridge-building between the components it connects.",
                "confidence": "high",
            })

    # Blocked perceptions (from structural_change)
    if isinstance(structural_change, dict):
        blocked = structural_change.get("blockages", {}).get("blocked_perception_sample", [])
        for pid in blocked:
            lb = label_for(pid, node_labels)
            hypotheses.append({
                "id": f"H4_{pid}",
                "type": "blockage_perception",
                "node_id": pid,
                "node_type": "perception",
                "label": lb,
                "metric": "no_path_to_information",
                "hypothesis": (
                    f"Perception '{lb}' ({pid}) has no path to any information node. "
                    f"This perception exists in a discursive silo — it cannot be informed, challenged, "
                    f"or influenced by the broader ecosystem's communication channels."
                ),
                "framework": "Discursive Cleavages / Ideological Polarization",
                "action": "Deploy an information channel targeting the sector/theme of this perception. Fund boundary-spanning initiatives.",
                "confidence": "high",
            })

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. LOCK-IN HYPOTHESES — policy monopoly / entrenched core
    # ═══════════════════════════════════════════════════════════════════════════
    print("--- Lock-in hypotheses ---")

    if isinstance(structural_change, dict):
        max_k = structural_change.get("graph_summary", {}).get("max_k_core", 0)
        frac_core = structural_change.get("graph_summary", {}).get("fraction_in_dense_core", 0)
        rob_gap = structural_change.get("path_dependency", {}).get("robustness_gap_targeted_vs_random", 0)

        # Identify which nodes are in the innermost k-core
        core_nodes = []
        if not readiness_nodes.empty:
            max_k_local = readiness_nodes["k_core"].max()
            core_nodes = readiness_nodes[readiness_nodes["k_core"] == max_k_local].to_dict("records")

        core_labels = [label_for(r.get("global_id", ""), node_labels) for r in core_nodes[:10]]
        core_types = [node_types.get(r.get("global_id", ""), "unknown") for r in core_nodes[:10]]
        core_summary = pd.Series(core_types).value_counts().to_dict() if core_types else {}

        hypotheses.append({
            "id": "H5_lockin",
            "type": "lockin_policy_monopoly",
            "node_id": "|".join([r.get("global_id", "") for r in core_nodes[:10]]),
            "node_type": str(core_summary),
            "label": " | ".join(core_labels[:5]),
            "metric": f"max_k_core={max_k}, fraction={frac_core:.3f}, robustness_gap={rob_gap:.4f}",
            "hypothesis": (
                f"The innermost k-core (k={max_k}) contains {len(core_nodes)} nodes ({frac_core*100:.1f}% of graph). "
                f"Types: {core_summary}. This resembles a policy monopoly (Sabatier & Jenkins-Smith, 1993) — "
                f"an entrenched coalition controlling resources and framing. "
                f"The robustness gap ({rob_gap:.4f}) indicates the system degrades faster under targeted removal "
                f"than random failure, confirming dependency on this core."
            ),
            "framework": "Policy Monopoly / Dominant Coalition (ACF + PET)",
            "action": (
                "Don't challenge the core directly. Fund venue-shopping: peripheral advocacy groups "
                "that can reframe the dominant issue image and shift debates to alternative forums. "
                "Identify secondary-belief alignments where peripheral coalitions can form tactical alliances."
            ),
            "confidence": "high",
        })

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. PLASTICITY HYPOTHESES — capacity for new connections
    # ═══════════════════════════════════════════════════════════════════════════
    print("--- Plasticity hypotheses ---")

    if isinstance(structural_change, dict):
        n_peripheral = structural_change.get("plasticity", {}).get("n_peripheral_nodes_deg_1_2", 0)
        n_merging = structural_change.get("plasticity", {}).get("n_links_that_merge_components", 0)
        n_high_conf = structural_change.get("plasticity", {}).get("n_gnn_high_conf_new_links", 0)

        hypotheses.append({
            "id": "H6_plasticity",
            "type": "plasticity_capacity",
            "node_id": "",
            "node_type": "",
            "label": f"{n_peripheral} peripheral nodes, {n_high_conf} high-confidence GNN links",
            "metric": f"peripheral={n_peripheral}, merging_links={n_merging}, high_conf={n_high_conf}",
            "hypothesis": (
                f"The network has {n_peripheral} peripheral nodes (degree 1-2) — spare capacity for rewiring. "
                f"There are {n_high_conf} high-confidence GNN-predicted links and {n_merging} that would merge components. "
                f"Adding these links could integrate stranded nodes without triggering systemic resistance."
            ),
            "framework": "Latent Collaborative Capacity (NPG)",
            "action": (
                f"Prioritize the {n_merging} component-merging links to reduce fragmentation. "
                f"Design joint initiatives that pair peripheral nodes with core actors."
            ),
            "confidence": "medium",
        })

        # GNN link recommendations — specific pairing hypotheses
        if not gnn_recs.empty:
            for _, row in gnn_recs.iterrows():
                src = str(row.get("source_global_id", ""))
                tgt = str(row.get("target_global_id", ""))
                src_lb = row.get("source_label", label_for(src, node_labels))
                tgt_lb = row.get("target_label", label_for(tgt, node_labels))
                prob = row.get("link_probability", 0)
                rec_type = row.get("recommendation_category", "unknown")

                hypotheses.append({
                    "id": f"H7_{src}_{tgt}",
                    "type": "plasticity_link_addition",
                    "node_id": f"{src} → {tgt}",
                    "node_type": f"{row.get('source_node_type', '')} → {row.get('target_node_type', '')}",
                    "label": f"{src_lb} ↔ {tgt_lb}",
                    "metric": f"link_probability={prob:.3f}, category={rec_type}",
                    "hypothesis": (
                        f"Linking '{src_lb}' to '{tgt_lb}' (probability={prob:.3f}) could "
                        f"{'merge disconnected components' if row.get('pair_kind') == 'bridge' else 'reinforce collaboration'}. "
                        f"This is a '{rec_type}' recommendation."
                    ),
                    "framework": "Latent Collaborative Capacity",
                    "action": "Design a joint initiative requiring partnership between these two entities.",
                    "confidence": "high" if prob > 0.9 else "medium",
                })

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. CLAIM-BASED HYPOTHESES — narrative interventions
    # ═══════════════════════════════════════════════════════════════════════════
    print("--- Claim-based hypotheses ---")

    if not claim_nodes.empty:
        # Claims grouped by value dimension
        vdim_groups = claim_nodes[claim_nodes["value_dimension"].notna() & (claim_nodes["value_dimension"] != "")]
        if not vdim_groups.empty:
            for vdim, group in vdim_groups.groupby("value_dimension"):
                claims_in_group = group["global_id"].tolist()
                # Find linked entities via claim edges
                if not claim_edges.empty:
                    linked_entities = claim_edges[
                        claim_edges["source_global_id"].isin(claims_in_group) &
                        claim_edges["edge_type"].str.contains("claim_about", na=False)
                    ]["target_global_id"].tolist()
                    linked_entities += claim_edges[
                        claim_edges["target_global_id"].isin(claims_in_group) &
                        claim_edges["edge_type"].str.contains("claim_about", na=False)
                    ]["source_global_id"].tolist()
                else:
                    linked_entities = []

                entity_labels = [label_for(e, node_labels) for e in linked_entities[:5]]
                entity_types = [node_types.get(e, "unknown") for e in linked_entities[:5]]

                n_surface = int((group["narrative_level"] == "surface").sum())
                n_implicit = int((group["narrative_level"] == "implicit").sum())

                hypotheses.append({
                    "id": f"H8_{vdim}",
                    "type": "narrative_cluster",
                    "node_id": ",".join(claims_in_group),
                    "node_type": f"claim ({vdim})",
                    "label": f"Narrative cluster: {vdim} ({len(claims_in_group)} claims)",
                    "metric": f"surface={n_surface}, implicit={n_implicit}, entities_linked={len(linked_entities)}",
                    "hypothesis": (
                        f"The '{vdim}' narrative cluster contains {len(claims_in_group)} claims "
                        f"({n_surface} surface, {n_implicit} implicit), linked to {len(linked_entities)} entities "
                        f"including {entity_labels[:3]}. "
                        f"{'This value dimension is dominant in explicit discourse but may have unstated premises.' if n_surface > n_implicit else 'Most claims are implicit — this value operates as an unstated assumption, making it harder to challenge.'}"
                    ),
                    "framework": "Advocacy Coalition Framework — Belief Hierarchy",
                    "action": (
                        f"Amplify '{vdim}' narratives through boundary-spanning communication. "
                        f"Link to matched funding streams. "
                        f"Monitor for narrative capture by dominant coalitions."
                    ),
                    "confidence": "medium",
                })

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. FINANCIAL-PERCEPTION BRIDGE
    # ═══════════════════════════════════════════════════════════════════════════
    print("--- Financial-perception bridge ---")

    bridge_rows = []

    if not claim_nodes.empty and not fin_diffusion.empty and not claim_edges.empty:
        # Map claims → entities → financial data
        claim_entity_map = []
        for _, row in claim_edges.iterrows():
            etype = str(row.get("edge_type", ""))
            if "claim_about" in etype:
                src = str(row.get("source_global_id", ""))
                tgt = str(row.get("target_global_id", ""))
                claim_gid = src if src.startswith("claim") else tgt
                entity_gid = tgt if src.startswith("claim") else src
                claim_entity_map.append({"claim_id": claim_gid, "entity_id": entity_gid})

        if claim_entity_map:
            cem = pd.DataFrame(claim_entity_map)

            # Merge claim metadata
            claim_info = claim_nodes[["global_id", "value_dimension", "narrative_level"]].copy()
            claim_info = claim_info.rename(columns={"global_id": "claim_id"})
            merged = cem.merge(claim_info, on="claim_id", how="left")

            # Merge financial data
            fin_info = fin_diffusion[["global_id", "label", "node_type", "investment_eur", "financial_bias"]].copy()
            fin_info = fin_info.rename(columns={"global_id": "entity_id"})
            merged = merged.merge(fin_info, on="entity_id", how="inner")

            if not merged.empty and "value_dimension" in merged.columns:
                # Cross-tabulation: value_dimension × financial metrics
                cross_tab = merged.groupby("value_dimension").agg(
                    n_claims=("claim_id", "nunique"),
                    n_entities=("entity_id", "nunique"),
                    total_investment_eur=("investment_eur", "sum"),
                    mean_investment_eur=("investment_eur", "mean"),
                    mean_financial_bias=("financial_bias", "mean"),
                    entity_labels=("label", lambda x: list(x.unique())[:5]),
                ).reset_index()

                cross_tab = cross_tab[cross_tab["value_dimension"].notna() & (cross_tab["value_dimension"] != "")]
                cross_tab = cross_tab.sort_values("total_investment_eur", ascending=False)

                # Write the bridge table
                write_frame(cross_tab, "financial_perception_bridge.csv")
                print(f"  Wrote financial_perception_bridge.csv ({len(cross_tab)} value dimensions)")

                # Generate hypotheses from cross-tab
                if len(cross_tab) > 1:
                    top_dim = cross_tab.iloc[0]
                    bottom_dim = cross_tab.iloc[-1]
                    hypotheses.append({
                        "id": "H9_finance_value_gap",
                        "type": "financial_perception_gap",
                        "node_id": "",
                        "node_type": f"{top_dim['value_dimension']} vs {bottom_dim['value_dimension']}",
                        "label": f"Funding gap: {bottom_dim['value_dimension']} narratives are under-resourced",
                        "metric": (
                            f"top={top_dim['value_dimension']}: €{top_dim['total_investment_eur']:,.0f} "
                            f"({top_dim['n_claims']} claims, {top_dim['n_entities']} entities) | "
                            f"bottom={bottom_dim['value_dimension']}: €{bottom_dim['total_investment_eur']:,.0f} "
                            f"({bottom_dim['n_claims']} claims, {bottom_dim['n_entities']} entities)"
                        ),
                        "hypothesis": (
                            f"Claim value dimension '{bottom_dim['value_dimension']}' is linked to only "
                            f"€{bottom_dim['total_investment_eur']:,.0f} in funding ({bottom_dim['n_entities']} entities), "
                            f"while '{top_dim['value_dimension']}' commands €{top_dim['total_investment_eur']:,.0f}. "
                            f"This may represent a structural misalignment between narrative priorities and resource allocation."
                        ),
                        "framework": "Material vs Purposive Beliefs (ACF)",
                        "action": (
                            f"Reallocate funding to bridge '{bottom_dim['value_dimension']}' initiatives. "
                            f"Use narrative amplification to make '{bottom_dim['value_dimension']}' more visible to funders."
                        ),
                        "confidence": "medium",
                    })

            # Perception-finance link
            # Check if perceptions with high financial_bias cluster around specific value dimensions
            if not merged.empty and "financial_bias" in merged.columns:
                bias_by_vdim = merged.groupby("value_dimension")["financial_bias"].mean()
                high_bias = bias_by_vdim[bias_by_vdim > 0.5]
                low_bias = bias_by_vdim[bias_by_vdim < -0.5]
                if not high_bias.empty:
                    for vdim, bias in high_bias.items():
                        bridge_rows.append({
                            "value_dimension": vdim,
                            "financial_bias": round(bias, 3),
                            "bias_direction": "finance_advantaged",
                            "implication": "Narratives in this value dimension are linked to entities that receive more funding than their network centrality would predict.",
                        })
                if not low_bias.empty:
                    for vdim, bias in low_bias.items():
                        bridge_rows.append({
                            "value_dimension": vdim,
                            "financial_bias": round(bias, 3),
                            "bias_direction": "finance_disadvantaged",
                            "implication": "Narratives in this value dimension are linked to entities that receive less funding than their network centrality would predict.",
                        })

    # Narrative level × budget (surface vs implicit claims and their linked budgets)
    if not claim_nodes.empty and not claim_edges.empty:
        level_map = []
        for _, row in claim_edges.iterrows():
            etype = str(row.get("edge_type", ""))
            if "claim_about" in etype:
                src = str(row.get("source_global_id", ""))
                tgt = str(row.get("target_global_id", ""))
                claim_gid = src if src.startswith("claim") else tgt
                entity_gid = tgt if src.startswith("claim") else src
                level_map.append({"claim_id": claim_gid, "entity_id": entity_gid})

        if level_map:
            lm = pd.DataFrame(level_map)
            cn_lvl = claim_nodes[["global_id", "narrative_level", "value_dimension"]].rename(columns={"global_id": "claim_id"})
            lm = lm.merge(cn_lvl, on="claim_id", how="left")

            # Merge with value_leverage for associated_budget
            if not value_leverage.empty:
                vl_info = value_leverage[["global_id", "associated_budget"]].rename(columns={"global_id": "entity_id"})
                lm = lm.merge(vl_info, on="entity_id", how="inner")

            if not lm.empty and "narrative_level" in lm.columns:
                agg_dict = {"n_claims": ("claim_id", "nunique"), "n_entities": ("entity_id", "nunique")}
                if "associated_budget" in lm.columns:
                    agg_dict["total_budget"] = ("associated_budget", "sum")
                level_budget = lm.groupby(["narrative_level", "value_dimension"]).agg(**agg_dict).reset_index()
                write_frame(level_budget, "narrative_level_budget_crosstab.csv")
                print(f"  Wrote narrative_level_budget_crosstab.csv ({len(level_budget)} rows)")

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. RECOMMENDATIONS — ranked by impact
    # ═══════════════════════════════════════════════════════════════════════════
    print("--- Recommendations ---")

    # Derive top recommendations from hypotheses
    for h in hypotheses:
        if h["confidence"] == "high":
            recommendations.append({
                "id": h["id"],
                "title": h["label"],
                "action": h["action"],
                "framework": h["framework"],
                "type": h["type"],
                "priority": "high" if "blockage" in h["type"] or "lockin" in h["type"] else "medium",
            })

    # ═══════════════════════════════════════════════════════════════════════════
    # 8. WRITE OUTPUT
    # ═══════════════════════════════════════════════════════════════════════════
    output = {
        "platform_id": str(nodes_df["platform_id"].iloc[0]) if not nodes_df.empty and "platform_id" in nodes_df.columns else "unknown",
        "n_hypotheses": int(len(hypotheses)),
        "n_recommendations": int(len(recommendations)),
        "overall_assessment": _overall_assessment(hypotheses),
        "hypotheses": _to_native(hypotheses),
        "recommendations": _to_native(sorted(recommendations, key=lambda r: 0 if r["priority"] == "high" else 1)),
    }

    write_json(ANALYSIS_DIR / "structural_hypotheses.json", output)
    print(f"\nWrote {len(hypotheses)} hypotheses, {len(recommendations)} recommendations")
    print(f"Output: {ANALYSIS_DIR / 'structural_hypotheses.json'}")
    print("Done.")


def load_json(path: Path) -> dict | list | None:
    if path.exists() and path.stat().st_size > 2:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _to_native(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    try:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except TypeError:
        pass
    return obj


def _overall_assessment(hypotheses: list[dict]) -> str:
    n_lockin = sum(1 for h in hypotheses if "lockin" in h.get("type", ""))
    n_blockage = sum(1 for h in hypotheses if "blockage" in h.get("type", ""))
    n_leverage = sum(1 for h in hypotheses if "leverage" in h.get("type", ""))
    high_conf = sum(1 for h in hypotheses if h.get("confidence") == "high")

    lines = [
        f"Generated {len(hypotheses)} hypotheses across {n_leverage} leverage, {n_blockage} blockage, and {n_lockin} lock-in dimensions.",
        f"High-confidence hypotheses: {high_conf}.",
        "Bottom line: ",
    ]
    if n_blockage > n_leverage:
        lines.append("The network has more structural blockages than leverage points — change requires new bridges, not just strengthening existing ones.")
    elif n_lockin > 2:
        lines.append("The network shows signs of policy monopoly — peripheral venue-shopping is recommended over direct challenge.")
    else:
        lines.append("The network is structurally open — targeted link addition and brokerage strengthening can enable change.")

    return " ".join(lines)


if __name__ == "__main__":
    main()

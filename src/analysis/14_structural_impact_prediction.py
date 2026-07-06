"""Narrative impact of GNN-predicted links + node addition proposals.

Flow:
  1. Load GNN link recommendations (structural holes from graph topology)
  2. For each recommended link:
     a. BFS source and target neighborhoods → claim/perception nodes + value dimensions
     b. Counterfactual: temporarily add edge → Δ narrative reachability
     c. Classify bridge type (learning_bridge / coalition_reinforcement / cleavage_breach / structural)
     d. If endpoint is isolated (no narrative neighbors), propose a NODE addition instead
  3. Detect narrative gaps across the graph (underfunded values, perception-void components)
  4. Output unified scored results

Outputs per platform:
  - narrative_impact_predictions.csv — all candidates (edges + nodes) with impact scores
  - narrative_impact_report.json — summary with findings
  - narrative_impact_dashboard.json — pre-formatted for Streamlit rendering

Usage:
    set KTOOL_PLATFORM_ID=173 & set KTOOL_OUTPUT_SUBDIR=test
    python src/analysis/14_structural_impact_prediction.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from graph_utils import (
    ANALYSIS_DIR,
    ANALYTICS_DIR,
    DATA_DIR,
    build_graph,
    read_csv_safe,
    simple_graph,
    write_frame,
    write_json,
)


def load_csv_safe(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 2:
        return pd.read_csv(path)
    return pd.DataFrame()


VALUE_LABELS = {
    "cultural_identity": "Cultural Identity",
    "social_justice": "Social Justice",
    "collaboration": "Collaboration",
    "innovation_drive": "Innovation Drive",
    "evidence_based": "Evidence-Based",
    "community_autonomy": "Community Autonomy",
    "austerity_scarcity": "Austerity & Scarcity",
}

ISSUE_COLORS = {
    "fragility": "#e74c3c",
    "isolation": "#e67e22",
    "kcore_exclusion": "#f39c12",
    "value_underfunding": "#2ecc71",
    "narrative_cleavage": "#3498db",
    "perception_isolation": "#9b59b6",
}


def load_graph_and_registry() -> tuple[nx.Graph, dict, dict]:
    G_multi = build_graph(directed=False)
    G = simple_graph(G_multi) if hasattr(G_multi, "to_undirected") else nx.Graph(G_multi)

    claim_edges = load_csv_safe(ANALYSIS_DIR / "narrative_layers" / "claim_edges.csv")
    for _, row in claim_edges.iterrows():
        s, t = str(row["source_global_id"]), str(row["target_global_id"])
        if G.has_node(s) and G.has_node(t):
            G.add_edge(s, t, weight=row.get("weight", 0.8), edge_origin="claim_edge")

    nodes_df = read_csv_safe(ANALYTICS_DIR / "nodes.csv")
    if nodes_df.empty:
        nodes_df = read_csv_safe(DATA_DIR / "nodes.csv")

    reg: dict[str, dict] = {}
    for _, row in nodes_df.iterrows():
        gid = str(row["global_id"])
        reg[gid] = {
            "node_type": str(row["node_type"]),
            "label": str(row.get("label", "") or ""),
            "value_dimension": "",
            "narrative_level": "",
            "verb": "",
            "description": str(row.get("description", "") or ""),
        }

    claim_nodes = load_csv_safe(ANALYSIS_DIR / "narrative_layers" / "claim_nodes.csv")
    for _, row in claim_nodes.iterrows():
        gid = str(row["global_id"])
        if gid in reg:
            reg[gid]["value_dimension"] = str(row.get("value_dimension", "") or "")
            reg[gid]["narrative_level"] = str(row.get("narrative_level", "") or "")
            reg[gid]["verb"] = str(row.get("verb", "") or "")
            reg[gid]["description"] = str(row.get("description", "") or "")

    for node in G.nodes():
        if node not in reg:
            reg[node] = {"node_type": "unknown", "label": node, "value_dimension": "",
                         "narrative_level": "", "verb": "", "description": ""}
        elif not reg[node]["label"]:
            reg[node]["label"] = node

    vd_gids = {}
    for gid, info in reg.items():
        vd = info.get("value_dimension", "")
        if vd and vd != "nan":
            vd_gids.setdefault(vd, []).append(gid)

    return G, reg, vd_gids


def bfs_narrative(
    G: nx.Graph, start: str, reg: dict, max_hops: int = 3, exclude: set | None = None,
) -> tuple[set[str], dict[str, str], set[str]]:
    exclude = exclude or set()
    if start not in G:
        return set(), {}, set()
    seen = {start} | exclude
    frontier = {start}
    reached_nodes: set[str] = set()
    for _ in range(max_hops):
        nf: set[str] = set()
        for node in frontier:
            for nb in G.neighbors(node):
                if nb in seen:
                    continue
                seen.add(nb)
                nf.add(nb)
                info = reg.get(nb, {})
                if info.get("node_type") in ("perception", "claim"):
                    reached_nodes.add(nb)
        frontier = nf
        if not frontier:
            break
    vd_map = {}
    for nid in reached_nodes:
        vd = reg.get(nid, {}).get("value_dimension", "")
        if vd and vd != "nan":
            vd_map[nid] = vd
    return reached_nodes, vd_map, seen


def format_vd_set(vd_set: set[str]) -> str:
    return ", ".join(sorted(VALUE_LABELS.get(v, v) for v in vd_set if v and v != "nan")) or "none"


def format_nodes(nids: set[str], reg: dict, limit: int = 5) -> str:
    if not nids:
        return "none"
    items = []
    for nid in sorted(nids)[:limit]:
        info = reg.get(nid, {})
        label = info.get("label", nid)[:40]
        vd = info.get("value_dimension", "")
        vd_label = VALUE_LABELS.get(vd, vd) if vd and vd != "nan" else ""
        vd_part = f" [{vd_label}]" if vd_label else ""
        items.append(f"{label}{vd_part}")
    if len(nids) > limit:
        items.append(f"+{len(nids) - limit} more")
    return "; ".join(items)


def classify_bridge_type(
    src_vds: dict[str, str], tgt_vds: dict[str, str],
    new_total: set[str], new_vds: set[str],
) -> tuple[str, str]:
    src_vd_set = set(src_vds.values()) if src_vds else set()
    tgt_vd_set = set(tgt_vds.values()) if tgt_vds else set()
    src_vd_set = {v for v in src_vd_set if v and v != "nan"}
    tgt_vd_set = {v for v in tgt_vd_set if v and v != "nan"}

    if not new_total:
        return "structural_only", "No new perception/narrative pathways open — this link is purely structural."
    if new_vds and new_vds - (src_vd_set | tgt_vd_set):
        return "cleavage_breach", f"Breaches narrative cleavage — unlocks value dimensions ({format_vd_set(new_vds)}) that exist in NEITHER side's current neighborhood."
    if new_vds and new_vds & (src_vd_set | tgt_vd_set):
        return "learning_bridge", f"Learning bridge — unlocks {format_vd_set(new_vds)} across previously separate narrative spaces."
    if src_vd_set and tgt_vd_set and not (src_vd_set & tgt_vd_set):
        return "learning_bridge", f"Connects two different value spaces ({format_vd_set(src_vd_set)} vs {format_vd_set(tgt_vd_set)}) — creates cross-belief dialogue."
    if src_vd_set and tgt_vd_set:
        return "coalition_reinforcement", f"Both sides share value dimensions ({format_vd_set(src_vd_set & tgt_vd_set)}). Reinforces existing narrative coalition."
    return "structural_only", "Neither side has identifiable value dimensions. Primarily structural."


def simulate_edge_impact(
    G: nx.Graph, src: str, tgt: str, reg: dict,
) -> dict:
    src_reached, src_vds, _ = bfs_narrative(G, src, reg)
    tgt_reached, tgt_vds, _ = bfs_narrative(G, tgt, reg)

    G_temp = G.copy()
    G_temp.add_edge(src, tgt)
    src_post, src_post_vds, _ = bfs_narrative(G_temp, src, reg)
    tgt_post, tgt_post_vds, _ = bfs_narrative(G_temp, tgt, reg)

    new_src = src_post - src_reached
    new_tgt = tgt_post - tgt_reached
    new_total = new_src | new_tgt

    src_vd_set = {v for v in src_vds.values() if v and v != "nan"} if src_vds else set()
    tgt_vd_set = {v for v in tgt_vds.values() if v and v != "nan"} if tgt_vds else set()
    new_vds_from_src = {v for v in src_post_vds.values() if v and v != "nan"} - src_vd_set if src_post_vds else set()
    new_vds_from_tgt = {v for v in tgt_post_vds.values() if v and v != "nan"} - tgt_vd_set if tgt_post_vds else set()
    all_new_vds = new_vds_from_src | new_vds_from_tgt
    only_src_vds = src_vd_set - tgt_vd_set
    only_tgt_vds = tgt_vd_set - src_vd_set
    bridge_type, bridge_note = classify_bridge_type(src_vds, tgt_vds, new_total, all_new_vds)

    return {
        "src_reached": src_reached,
        "tgt_reached": tgt_reached,
        "src_vds": src_vds,
        "tgt_vds": tgt_vds,
        "new_total": new_total,
        "all_new_vds": all_new_vds,
        "only_src_vds": only_src_vds,
        "only_tgt_vds": only_tgt_vds,
        "bridge_type": bridge_type,
        "bridge_note": bridge_note,
        "src_vd_set": src_vd_set,
        "tgt_vd_set": tgt_vd_set,
    }


def propose_node_addition(
    node_id: str, reg: dict, vd_gids: dict, G: nx.Graph,
) -> list[dict]:
    """For an isolated or narratively-poor node, propose what new node would help."""
    proposals = []
    info = reg.get(node_id, {})
    node_type = info.get("node_type", "unknown")
    label = info.get("label", node_id)

    reached, vds, _ = bfs_narrative(G, node_id, reg)
    vd_set = {v for v in vds.values() if v and v != "nan"}

    # 1. If node has no claim neighbors → propose a claim node
    reached_types = [reg.get(n, {}).get("node_type", "") for n in reached]
    claim_count = reached_types.count("claim")
    if claim_count == 0:
        # Find which value dimensions are UNDERPEPRESENTED in this node's type vicinity
        nearby = list(G.neighbors(node_id)) if node_id in G else []
        nearby_types = [reg.get(n, {}).get("node_type", "") for n in nearby]
        agent_or_proj_nearby = any(t in ("agent", "project") for t in nearby_types)
        # Find most under-represented value dimension
        vd_counts = {vd: len(gids) for vd, gids in vd_gids.items()} if vd_gids else {}
        rarest_vd = min(vd_counts, key=vd_counts.get) if vd_counts else "community_autonomy"
        rarest_vd_label = VALUE_LABELS.get(rarest_vd, rarest_vd)
        proposals.append({
            "proposal_type": "node_addition",
            "proposal_subtype": "claim_node",
            "anchor_node": node_id,
            "anchor_label": label,
            "proposed_value_dimension": rarest_vd,
            "rationale": f"{label} ({node_type}) has no claim nodes nearby. "
                         f"Adding a claim expressing '{rarest_vd_label}' (currently {vd_counts.get(rarest_vd, 0)} claims in graph) "
                         f"would give this node a narrative voice in an under-represented value space.",
        })

    # 2. If node's nearby claims lack a specific value dimension → propose filling that gap
    present_vds = vd_set
    all_vds = set(VALUE_LABELS.keys())
    missing_vds = all_vds - present_vds
    if missing_vds and claim_count > 0:
        rarest_missing = min(missing_vds, key=lambda v: len(vd_gids.get(v, []))) if missing_vds and vd_gids else None
        if rarest_missing:
            proposals.append({
                "proposal_type": "node_addition",
                "proposal_subtype": "claim_value_gap",
                "anchor_node": node_id,
                "anchor_label": label,
                "proposed_value_dimension": rarest_missing,
                "rationale": f"{label}'s neighborhood expresses {format_vd_set(present_vds)} but lacks '{VALUE_LABELS.get(rarest_missing, rarest_missing)}'. "
                             f"Adding a claim in this value dimension would diversify the narrative portfolio.",
            })

    # 3. If component has no perception node → propose one
    comp_membership = load_csv_safe(ANALYSIS_DIR / "component_membership.csv")
    if not comp_membership.empty:
        node_row = comp_membership[comp_membership["global_id"] == node_id]
        if not node_row.empty:
            comp_id = str(node_row.iloc[0]["component_id"])
            comp_nodes = comp_membership[comp_membership["component_id"] == comp_id]["global_id"].tolist()
            has_perception = any(reg.get(n, {}).get("node_type") == "perception" for n in comp_nodes)
            if not has_perception:
                proposals.append({
                    "proposal_type": "node_addition",
                    "proposal_subtype": "perception_node",
                    "anchor_node": node_id,
                    "anchor_label": label,
                    "proposed_value_dimension": "",
                    "rationale": f"The component containing {label} has no perception nodes. "
                                 f"Adding a perception node would give this community a voice in the perception space.",
                })

    return proposals


def main() -> None:
    print("=" * 70)
    print("NARRATIVE IMPACT OF GNN-PREDICTED LINKS")
    print("=" * 70)

    # ── Build graph ──
    print("\nBuilding graph + registry...")
    G, reg, vd_gids = load_graph_and_registry()
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Registry: {len(reg)} nodes")
    print(f"  Value dimension groups: {len(vd_gids)}")

    # ── Structural data ──
    comp = load_csv_safe(ANALYSIS_DIR / "component_membership.csv")
    art = load_csv_safe(ANALYSIS_DIR / "articulation_points.csv")
    kcore = load_csv_safe(ANALYSIS_DIR / "kcore_membership.csv")
    cent = load_csv_safe(ANALYSIS_DIR / "node_centrality.csv")

    art_nodes = set(art["global_id"].astype(str)) if not art.empty else set()
    comp_map = dict(zip(comp["global_id"].astype(str), comp["component_id"])) if not comp.empty else {}
    kcore_map = dict(zip(kcore["global_id"].astype(str), kcore["k_core_number"])) if not kcore.empty else {}
    deg_map = dict(zip(cent["global_id"].astype(str), cent["degree"])) if not cent.empty else {}
    bt_map = dict(zip(cent["global_id"].astype(str), cent["betweenness_centrality"])) if not cent.empty else {}

    # ── Financial-perception bridge ──
    fin_bridge = load_csv_safe(ANALYSIS_DIR / "financial_perception_bridge.csv")
    budget_by_vd: dict[str, float] = {}
    if not fin_bridge.empty and "budget" in fin_bridge.columns:
        for _, row in fin_bridge.iterrows():
            vd = str(row.get("value_dimension", "")).strip()
            b = float(row.get("budget", 0))
            if vd and vd != "nan":
                budget_by_vd[vd] = budget_by_vd.get(vd, 0) + b

    # ── Load GNN recommendations ──
    gnn_recs = load_csv_safe(ANALYSIS_DIR / "gnn" / "gnn_link_recommendations.csv")
    print(f"\nGNN recommendations: {len(gnn_recs)}")
    for _, rec in gnn_recs.iterrows():
        s, t = str(rec["source_global_id"]), str(rec["target_global_id"])
        sl = reg.get(s, {}).get("label", s)
        tl = reg.get(t, {}).get("label", t)
        prob = float(rec.get("link_probability", 0))
        print(f"  {sl} → {tl}  [p={prob:.3f}]")

    # ── Process each recommendation ──
    print("\n" + "=" * 70)
    print("PROCESSING GNN RECOMMENDATIONS")
    print("=" * 70)

    all_edge_candidates: list[dict] = []
    all_node_proposals: list[dict] = []

    for idx, rec in gnn_recs.iterrows():
        src = str(rec["source_global_id"])
        tgt = str(rec["target_global_id"])
        sl = reg.get(src, {}).get("label", src)
        tl = reg.get(tgt, {}).get("label", tgt)
        st = reg.get(src, {}).get("node_type", "?")
        tt = reg.get(tgt, {}).get("node_type", "?")
        prob = float(rec.get("link_probability", 0))
        category = str(rec.get("recommendation_category", ""))
        rationale = str(rec.get("rationale", ""))

        G_temp = G.copy()
        sim = simulate_edge_impact(G_temp, src, tgt, reg)

        src_vd_set = sim["src_vd_set"]
        tgt_vd_set = sim["tgt_vd_set"]

        # ── Structural context ──
        src_comp = comp_map.get(src, "?")
        tgt_comp = comp_map.get(tgt, "?")
        merges_components = src_comp != tgt_comp and src_comp != "?" and tgt_comp != "?"
        src_deg = G.degree(src) if src in G else 0
        tgt_deg = G.degree(tgt) if tgt in G else 0
        src_art = src in art_nodes
        tgt_art = tgt in art_nodes
        src_k = kcore_map.get(src, 0)
        tgt_k = kcore_map.get(tgt, 0)
        src_bt = bt_map.get(src, 0.0)
        tgt_bt = bt_map.get(tgt, 0.0)

        # ── Score ──
        new_count = len(sim["new_total"])
        new_vd_count = len(sim["all_new_vds"])
        bridge_scores = {"learning_bridge": 1.0, "coalition_reinforcement": 0.6, "cleavage_breach": 0.8, "structural_only": 0.2}
        bridge_score = bridge_scores.get(sim["bridge_type"], 0.2)
        merge_score = 0.3 if merges_components else -0.1
        deg_score = min(1.0, (src_deg + tgt_deg) / 20.0)  # higher degree = more potential for diffusion
        art_score = 0.2 if src_art or tgt_art else 0.0
        narrative_score = min(1.0, new_count / 30.0) * 0.5 + min(1.0, new_vd_count / 5.0) * 0.5
        composite = (
            narrative_score * 0.40 +
            bridge_score * 0.25 +
            merge_score * 0.15 +
            deg_score * 0.10 +
            art_score * 0.10
        )

        impact = {
            "recommendation_source": "gnn_link_prediction",
            "intervention_type": "edge_addition",
            "source_global_id": src,
            "target_global_id": tgt,
            "source_label": sl,
            "target_label": tl,
            "source_node_type": st,
            "target_node_type": tt,
            "link_probability": round(prob, 4),
            "gnn_rationale": rationale,
            "source_perception_claim_count": len(sim["src_reached"]),
            "target_perception_claim_count": len(sim["tgt_reached"]),
            "source_vds": format_vd_set(src_vd_set),
            "target_vds": format_vd_set(tgt_vd_set),
            "only_source_vds": format_vd_set(sim["only_src_vds"]),
            "only_target_vds": format_vd_set(sim["only_tgt_vds"]),
            "new_narrative_pathways": new_count,
            "new_vds_unlocked": format_vd_set(sim["all_new_vds"]),
            "bridge_type": sim["bridge_type"],
            "bridge_note": sim["bridge_note"],
            "source_degree": src_deg,
            "target_degree": tgt_deg,
            "source_kcore": src_k,
            "target_kcore": tgt_k,
            "source_betweenness": round(src_bt, 4),
            "target_betweenness": round(tgt_bt, 4),
            "source_is_articulation": src_art,
            "target_is_articulation": tgt_art,
            "merges_components": merges_components,
            "source_component": src_comp,
            "target_component": tgt_comp,
            "new_pathway_nodes": format_nodes(sim["new_total"], reg),
            "narrative_score": round(narrative_score, 4),
            "composite_governance_score": round(composite, 4),
        }
        all_edge_candidates.append(impact)

        print(f"\n  [{idx+1}] {sl} ({st}) → {tl} ({tt})  p={prob:.3f}")
        print(f"    Source neighborhood: {len(sim['src_reached'])} nodes | {impact['source_vds']}")
        print(f"    Target neighborhood: {len(sim['tgt_reached'])} nodes | {impact['target_vds']}")
        print(f"    New pathways: {new_count} | New VDs: {impact['new_vds_unlocked']}")
        print(f"    Bridge: {sim['bridge_type']} | Score: {composite:.4f}")
        print(f"    Merges components: {merges_components}")

        # ── Node proposals for isolated endpoints ──
        for nid in [src, tgt]:
            node_deg = G.degree(nid) if nid in G else 0
            node_nt = reg.get(nid, {}).get("node_type", "")
            reached_n, vds_n, _ = bfs_narrative(G, nid, reg)
            if node_deg <= 2 or len(reached_n) == 0:
                proposals = propose_node_addition(nid, reg, vd_gids, G)
                for p in proposals:
                    p["linked_edge_source"] = src
                    p["linked_edge_target"] = tgt
                    p["linked_edge_probability"] = prob
                    p["linked_edge_bridge_type"] = sim["bridge_type"]
                    p["linked_edge_rationale"] = f"GNN suggests linking {sl} ↔ {tl} (p={prob:.3f}) but {reg.get(nid, {}).get('label', nid)} has no narrative neighbors — a node addition would create a narrative footprint for this structural intervention."
                    all_node_proposals.append(p)

    # ── Detect narrative gaps ──
    print("\n" + "=" * 70)
    print("DETECTING NARRATIVE GAPS")
    print("=" * 70)

    if vd_gids and budget_by_vd:
        print("\n  Value underfunding gaps:")
        min_budget = min(budget_by_vd.values())
        for vd, b in sorted(budget_by_vd.items(), key=lambda x: x[1]):
            vdl = VALUE_LABELS.get(vd, vd)
            claim_count = len(vd_gids.get(vd, []))
            gap = b == min_budget
            marker = " <<< UNDERFUNDED" if gap else ""
            print(f"    {vdl:25s}: €{b:>8,.0f} | {claim_count} claims{marker}")
            if gap:
                # Find which projects could host a new claim in this dimension
                host_candidates = []
                for nid, info in reg.items():
                    if info.get("node_type") in ("project", "pilot", "prototype"):
                        host_candidates.append((nid, info.get("label", nid)))
                for hn, hl in host_candidates[:3]:
                    all_node_proposals.append({
                        "proposal_type": "gap_filling",
                        "proposal_subtype": "value_underfunding",
                        "anchor_node": hn,
                        "anchor_label": hl,
                        "proposed_value_dimension": vd,
                        "rationale": f"'{vdl}' has €{b:,.0f} budget (lowest across all value dimensions). "
                                     f"Adding a claim node expressing '{vdl}' near '{hl}' ({info.get('node_type', '')}) "
                                     f"would connect this underfunded narrative to a concrete initiative.",
                        "linked_edge_source": "",
                        "linked_edge_target": "",
                        "linked_edge_probability": 0.0,
                        "linked_edge_bridge_type": "",
                        "linked_edge_rationale": "",
                    })
    elif vd_gids:
        # Budget data not available — use claim frequency as proxy
        print("\n  Narrative representation gaps (no budget data, using claim frequency):")
        vd_counts = {vd: len(gids) for vd, gids in vd_gids.items()}
        min_count = min(vd_counts.values())
        for vd, cnt in sorted(vd_counts.items(), key=lambda x: x[1]):
            vdl = VALUE_LABELS.get(vd, vd)
            gap = cnt == min_count
            marker = " <<< UNDER-REPRESENTED" if gap else ""
            print(f"    {vdl:25s}: {cnt} claims{marker}")
            if gap:
                for nid, info in list(reg.items())[:5]:
                    if info.get("node_type") in ("project", "pilot", "prototype"):
                        all_node_proposals.append({
                            "proposal_type": "gap_filling",
                            "proposal_subtype": "value_under_representation",
                            "anchor_node": nid,
                            "anchor_label": info.get("label", nid),
                            "proposed_value_dimension": vd,
                            "rationale": f"'{vdl}' has only {cnt} claims (least represented). "
                                         f"Adding a claim near '{info.get('label', nid)}' would diversify the narrative landscape.",
                            "linked_edge_source": "",
                            "linked_edge_target": "",
                            "linked_edge_probability": 0.0,
                            "linked_edge_bridge_type": "",
                            "linked_edge_rationale": "",
                        })

    # Perception-void components
    print("\n  Perception-void components:")
    if not comp.empty:
        comp_perception_map: dict[str, bool] = {}
        for _, row in comp.iterrows():
            cid = str(row["component_id"])
            gid = str(row["global_id"])
            is_perc = reg.get(gid, {}).get("node_type") == "perception"
            if cid not in comp_perception_map:
                comp_perception_map[cid] = False
            if is_perc:
                comp_perception_map[cid] = True
        void_comps = [cid for cid, has_perc in comp_perception_map.items() if not has_perc]
        print(f"    Components without perception nodes: {len(void_comps)} / {len(comp_perception_map)}")
        # Propose perception node for largest void component
        if void_comps:
            comp_gids_map = comp.groupby("component_id")["global_id"].apply(list).to_dict()
            largest_void = max(void_comps, key=lambda c: len(comp_gids_map.get(c, [])))
            anchor = reg.get(comp_gids_map[largest_void][0], {}).get("label", comp_gids_map[largest_void][0]) if comp_gids_map.get(largest_void) else ""
            all_node_proposals.append({
                "proposal_type": "gap_filling",
                "proposal_subtype": "perception_void",
                "anchor_node": comp_gids_map[largest_void][0] if comp_gids_map.get(largest_void) else "",
                "anchor_label": anchor,
                "proposed_value_dimension": "",
                "rationale": f"Component '{largest_void}' has {len(comp_gids_map.get(largest_void, []))} nodes but zero perception nodes. "
                             f"Adding a perception node would give this structural cluster a voice in the perception space.",
                "linked_edge_source": "",
                "linked_edge_target": "",
                "linked_edge_probability": 0.0,
                "linked_edge_bridge_type": "",
                "linked_edge_rationale": "",
            })

    # ── Combine and score ──
    print("\n" + "=" * 70)
    print("SCORING AND OUTPUT")
    print("=" * 70)

    edge_df = pd.DataFrame(all_edge_candidates)
    node_df = pd.DataFrame(all_node_proposals) if all_node_proposals else pd.DataFrame()

    if not edge_df.empty:
        edge_df = edge_df.sort_values("composite_governance_score", ascending=False)
        edge_path = write_frame(edge_df, "narrative_impact_predictions.csv")
        print(f"\n  Edge candidates: {len(edge_df)} → {edge_path}")

    if not node_df.empty:
        node_path = write_frame(node_df, "narrative_impact_node_proposals.csv")
        print(f"  Node proposals: {len(node_df)} → {node_path}")

    # ── Report ──
    top_edges = []
    for _, row in edge_df.head(5).iterrows():
        top_edges.append({
            "intervention_type": "edge_addition",
            "source": row["source_global_id"],
            "target": row["target_global_id"],
            "source_label": row["source_label"],
            "target_label": row["target_label"],
            "bridge_type": row["bridge_type"],
            "new_narrative_pathways": int(row["new_narrative_pathways"]),
            "new_vds_unlocked": row["new_vds_unlocked"],
            "composite_governance_score": float(row["composite_governance_score"]),
            "bridge_note": row["bridge_note"],
        })

    top_nodes = []
    for _, row in node_df.head(5).iterrows() if not node_df.empty else []:
        top_nodes.append({
            "intervention_type": "node_addition",
            "proposal_subtype": row.get("proposal_subtype", ""),
            "anchor_node": row.get("anchor_node", ""),
            "anchor_label": row.get("anchor_label", ""),
            "proposed_value_dimension": row.get("proposed_value_dimension", ""),
            "rationale": row.get("rationale", ""),
        })

    # Value dimension summary
    vd_summary = {}
    if vd_gids:
        for vd, gids in sorted(vd_gids.items(), key=lambda x: -len(x[1])):
            vdl = VALUE_LABELS.get(vd, vd)
            budget = budget_by_vd.get(vd, 0)
            vd_summary[vdl] = {
                "claim_count": len(gids),
                "budget": budget,
                "underfunded": bool(budget_by_vd and budget == min(budget_by_vd.values())),
            }

    report = {
        "platform_id": os.environ.get("KTOOL_PLATFORM_ID", "?"),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "gnn_recommendations_analyzed": len(all_edge_candidates),
        "node_proposals_generated": len(all_node_proposals),
        "value_dimensions": vd_summary,
        "perception_void_components": len(void_comps) if comp.empty is False and 'void_comps' in dir() else 0,
        "top_recommendations": {
            "edges": top_edges,
            "nodes": top_nodes,
        },
    }
    write_json(ANALYSIS_DIR / "narrative_impact_report.json", report)
    print(f"  Report → narrative_impact_report.json")

    # ── Dashboard JSON (pre-formatted for Streamlit) ──
    dashboard = {"edge_candidates": [], "node_proposals": [], "vd_summary": []}

    for _, row in edge_df.head(10).iterrows():
        item = row.to_dict()
        item["color"] = ISSUE_COLORS.get(row.get("bridge_type", ""), "#95a5a6")
        item["key"] = f"edge_{len(dashboard['edge_candidates'])}"
        dashboard["edge_candidates"].append(item)

    for _, row in node_df.head(10).iterrows() if not node_df.empty else []:
        item = row.to_dict()
        item["color"] = "#9b59b6"
        item["key"] = f"node_{len(dashboard['node_proposals'])}"
        dashboard["node_proposals"].append(item)

    for vdl, info in vd_summary.items():
        dashboard["vd_summary"].append({"value_dimension": vdl, **info})

    write_json(ANALYSIS_DIR / "narrative_impact_dashboard.json", dashboard)
    print(f"  Dashboard JSON → narrative_impact_dashboard.json")
    print("\nDone.")


if __name__ == "__main__":
    main()

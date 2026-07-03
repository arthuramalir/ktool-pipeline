"""Build nodes.csv + edges.csv for Platform 10 (Chile COPOLAD).
Adapted for schema differences: investment field, channel_code linkage, Spanish listening.

Usage:
  set KTOOL_PLATFORM_ID=10 & set KTOOL_OUTPUT_SUBDIR=test
  python src/analysis/00_prepare_graph_tables_p10.py
"""

from __future__ import annotations
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "10")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
DATA_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR
ENTITIES_DIR = DATA_DIR / "entities"
RELS_DIR = DATA_DIR / "relationships"
ANALYTICS_DIR = DATA_DIR / "analytics"
ANALYSIS_DIR = DATA_DIR / "analysis"
NODES_CSV = ANALYTICS_DIR / "nodes.csv"
EDGES_CSV = ANALYTICS_DIR / "edges.csv"

ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

INITIATIVE_TYPES = {"project", "pilot", "prototype"}


def load_table(name, subdir="entities"):
    path = DATA_DIR / subdir / name
    if not path.exists():
        path = DATA_DIR / name
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def text_val(row, *cols, default=""):
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c])
    return default


def parse_ids(val):
    if pd.isna(val) or not str(val).strip():
        return []
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return [str(x).strip() for x in json.loads(s) if x]
        except Exception:
            pass
    return [x.strip() for x in s.split(",") if x.strip()]


def add_node(rows, gid, nid, ntype, label, desc, phase, source, extra=None):
    if not gid or str(gid) == "nan":
        return
    node = {
        "global_id": str(gid),
        "native_id": "" if pd.isna(nid) else nid,
        "node_type": ntype,
        "label": label or "Unnamed",
        "description": desc or "",
        "methodological_phase": phase,
        "platform_id": PLATFORM_ID,
        "source_table": source,
    }
    if extra:
        for k, v in extra.items():
            if k not in node and not pd.isna(v):
                node[k] = v
    rows.append(node)


def add_edge(rows, source, target, etype, family, phase, directed, evidence, extra=None):
    if not source or not target or str(source) == "nan" or str(target) == "nan":
        return
    edge = {
        "source_global_id": str(source),
        "target_global_id": str(target),
        "edge_type": etype,
        "edge_family": family,
        "methodological_phase": phase,
        "directed": directed,
        "evidence_source": evidence,
        "platform_id": PLATFORM_ID,
        "is_ai_generated": False,
        "edge_origin": "source_data",
        "generated_by": "00_prepare_graph_tables_p10.py",
        "inference_method": "",
    }
    if extra:
        for k, v in extra.items():
            if k not in edge and not pd.isna(v):
                edge[k] = v
    rows.append(edge)


# ============================================================
# BUILD NODES
# ============================================================
def build_nodes():
    rows = []

    # --- Agents (123) ---
    agents = load_table("agents.csv")
    for _, r in agents.iterrows():
        extra = r.to_dict()
        # Preserve investment, type, people_involved as node attributes
        add_node(rows, f"agent_{r['id']}", r.get("id"), "agent",
                 text_val(r, "name"), text_val(r, "description"),
                 "mapping", "agents.csv", extra=extra)

    # --- Initiatives (projects, pilots, prototypes) ---
    for ntype, fname in [("project", "projects.csv"), ("pilot", "pilots.csv"), ("prototype", "prototypes.csv")]:
        df = load_table(fname)
        for _, r in df.iterrows():
            gid = text_val(r, "initiative_global_id", default=f"{ntype}_{r.get('native_id', r.get('id'))}")
            extra = r.to_dict()
            add_node(rows, gid, r.get("native_id"), ntype,
                     text_val(r, "title", "name"), text_val(r, "description"),
                     "mapping/co-creation", fname, extra=extra)

    # --- Information (567 Spanish quotes) ---
    info = load_table("informations.csv")
    for _, r in info.iterrows():
        gid = text_val(r, "global_id", default=f"information_{r.get('native_id', r.get('id'))}")
        extra = r.to_dict()
        add_node(rows, gid, r.get("native_id"), "information",
                 text_val(r, "name", default="Information"), text_val(r, "quote", "description"),
                 "listening", "informations.csv", extra=extra)

    # --- Channels (340) ---
    channels = load_table("channels.csv")
    for _, r in channels.iterrows():
        gid = text_val(r, "global_id", default=f"channel_{r.get('native_id', r.get('id'))}")
        extra = r.to_dict()
        add_node(rows, gid, r.get("native_id"), "channel",
                 text_val(r, "name", default="Channel"), text_val(r, "description"),
                 "listening", "channels.csv", extra=extra)

    # --- Perceptions (7) ---
    perceptions = load_table("perceptions.csv")
    for _, r in perceptions.iterrows():
        extra = r.to_dict()
        add_node(rows, f"perception_{r['id']}", r.get("id"), "perception",
                 text_val(r, "name", default="Perception"), text_val(r, "quote", "description"),
                 "listening", "perceptions.csv", extra=extra)

    # --- Challenges (35) ---
    challenges = load_table("challenges.csv")
    for _, r in challenges.iterrows():
        extra = r.to_dict()
        add_node(rows, f"challenge_{r['id']}", r.get("id"), "challenge",
                 text_val(r, "name"), text_val(r, "description"),
                 "listening", "challenges.csv", extra=extra)

    # --- Thematic Areas (6) ---
    themes = load_table("thematic_areas.csv")
    for _, r in themes.iterrows():
        extra = r.to_dict()
        add_node(rows, f"theme_{r['id']}", r.get("id"), "theme",
                 text_val(r, "name"), text_val(r, "description"),
                 "mapping", "thematic_areas.csv", extra=extra)

    # --- Values (from information-value if available) ---
    info_values = load_table("values.csv")
    for _, r in info_values.iterrows():
        extra = r.to_dict()
        add_node(rows, f"value_{r.get('id', '?')}", r.get("id"), "value",
                 text_val(r, "name", "value", default="Value"), text_val(r, "description"),
                 "listening", "values.csv", extra=extra)

    return pd.DataFrame(rows).drop_duplicates(subset="global_id")


# ============================================================
# BUILD EDGES
# ============================================================
def build_edges(nodes_df):
    rows = []
    valid_ids = set(nodes_df["global_id"].astype(str))

    # --- Agent → Initiative (from p10_{type}_lead_agent_links / p10_{type}_agent_links) ---
    # Extract script uses singular base_name: project, pilot, protot
    for init_base, init_type in [("project", "project"), ("pilot", "pilot"), ("protot", "prototype")]:
        for prefix, role in [("lead_agent", "leads"), ("agent", "participates_in")]:
            fname = f"p10_{init_base}_{prefix}_links.csv"
            df = load_table(fname, "relationships")
            if df.empty:
                continue
            for _, r in df.iterrows():
                init_gid = r.get("initiative_global_id")
                aid = r.get("agent_id")
                if pd.isna(aid) or pd.isna(init_gid):
                    continue
                agid = f"agent_{int(aid)}"
                if agid in valid_ids and str(init_gid) in valid_ids:
                    etype = role if role == "leads" else "participates_in"
                    add_edge(rows, agid, str(init_gid), etype, "declared_relational",
                             "mapping", False, f"agent {role} {init_type}")

    # --- Information → Channel (from link_information_channels.csv) ---
    link = load_table("link_information_channels.csv", "relationships")
    for _, r in link.iterrows():
        info_id = r.get("information_id")
        ch_id = r.get("channel_id")
        if pd.isna(info_id) or pd.isna(ch_id):
            continue
        info_gid = f"information_{int(info_id)}"
        ch_gid = f"channel_{int(ch_id)}"
        if info_gid in valid_ids and ch_gid in valid_ids:
            add_edge(rows, info_gid, ch_gid, "information_originates_from_channel", "listening",
                     "listening", True, "channel_code relation")

    # --- Information → Challenge (from p10_information_challenge_links) ---
    ic_links = load_table("p10_information_challenge_links.csv", "relationships")
    for _, r in ic_links.iterrows():
        info_id = r.get("information_id")
        ch_id = r.get("challenge_id")
        if not pd.isna(info_id) and not pd.isna(ch_id):
            info_gid = f"information_{int(info_id)}"
            ch_gid = f"challenge_{int(ch_id)}"
            if info_gid in valid_ids and ch_gid in valid_ids:
                add_edge(rows, info_gid, ch_gid, "information_relates_to_challenge", "interpretive",
                         "listening", False, "information-challenge link from p10 relink")

    # --- Information → Values (from information_value_links.csv) ---
    val_links = load_table("information_value_links.csv", "relationships")
    for _, r in val_links.iterrows():
        info_id = r.get("information_id")
        val_id = r.get("information_value_id")
        if pd.isna(info_id) or pd.isna(val_id):
            continue
        info_gid = f"information_{int(info_id)}"
        val_gid = f"value_{int(val_id)}"
        if info_gid in valid_ids and val_gid in valid_ids:
            add_edge(rows, info_gid, val_gid, "information_expresses_value", "interpretive",
                     "listening", False, "information-value link")

    # --- Perception → Challenge (from perception_challenge_links.csv) ---
    pc_links = load_table("perception_challenge_links.csv", "relationships")
    for _, r in pc_links.iterrows():
        p_id = r.get("perception_id")
        c_id = r.get("challenge_id")
        if not pd.isna(p_id) and not pd.isna(c_id):
            p_gid = f"perception_{int(p_id)}"
            c_gid = f"challenge_{int(c_id)}"
            if p_gid in valid_ids and c_gid in valid_ids:
                add_edge(rows, p_gid, c_gid, "perception_reveals_challenge", "interpretive",
                         "listening", False, "perception-challenge link")

    # --- Initiative → Perception (from p10_{type}_perception_links) ---
    for init_base in ["project", "pilot", "protot"]:
        fname = f"p10_{init_base}_perception_links.csv"
        df = load_table(fname, "relationships")
        if df.empty:
            continue
        for _, r in df.iterrows():
            init_gid = r.get("initiative_global_id")
            pid = r.get("perception_id")
            if pd.isna(init_gid) or pd.isna(pid):
                continue
            p_gid = f"perception_{int(pid)}"
            if str(init_gid) in valid_ids and p_gid in valid_ids:
                add_edge(rows, str(init_gid), p_gid, "initiative_has_perception", "interpretive",
                         "listening", False, "initiative-perception link from p10 relink")

    # --- Value → Perception (from value_perception_links.csv) ---
    vp_links = load_table("value_perception_links.csv", "relationships")
    for _, r in vp_links.iterrows():
        v_id = r.get("value_id")
        p_id = r.get("perception_id")
        if not pd.isna(v_id) and not pd.isna(p_id):
            v_gid = f"value_{int(v_id)}"
            p_gid = f"perception_{int(p_id)}"
            if v_gid in valid_ids and p_gid in valid_ids:
                add_edge(rows, v_gid, p_gid, "value_related_to_perception", "interpretive",
                         "sensemaking", False, "value-perception link")

    # --- Initiative → Thematic area (from p10_{type}_theme_links) ---
    for init_base in ["project", "pilot", "protot"]:
        fname = f"p10_{init_base}_theme_links.csv"
        df = load_table(fname, "relationships")
        if df.empty:
            continue
        for _, r in df.iterrows():
            init_gid = r.get("initiative_global_id")
            tid = r.get("thematic_area_id")
            if pd.isna(init_gid) or pd.isna(tid):
                continue
            theme_gid = f"theme_{int(tid)}"
            if str(init_gid) in valid_ids and theme_gid in valid_ids:
                add_edge(rows, str(init_gid), theme_gid, "initiative_addresses_theme", "declared_relational",
                         "mapping", False, "initiative-theme link from p10 relink")

    # --- Agent → Thematic area (from topics_thematic_areas on agents) ---
    for _, r in load_table("agents.csv").iterrows():
        agid = f"agent_{r['id']}"
        if agid not in valid_ids:
            continue
        tids = parse_ids(r.get("topic_ids", ""))
        for tid in tids:
            theme_gid = f"theme_{tid}"
            if theme_gid in valid_ids:
                add_edge(rows, agid, theme_gid, "agent_works_on_theme", "declared_relational",
                         "mapping", False, "agent-theme link")

    # --- Information → Thematic area (from p10_information_theme_links) ---
    it_links = load_table("p10_information_theme_links.csv", "relationships")
    for _, r in it_links.iterrows():
        info_id = r.get("information_id")
        tid = r.get("thematic_area_id")
        if not pd.isna(info_id) and not pd.isna(tid):
            info_gid = f"information_{int(info_id)}"
            theme_gid = f"theme_{int(tid)}"
            if info_gid in valid_ids and theme_gid in valid_ids:
                add_edge(rows, info_gid, theme_gid, "information_addresses_theme", "interpretive",
                         "listening", False, "information-theme link from p10 relink")

    # --- Challenge → Information (from p10_challenge_quote_links) ---
    cq_links = load_table("p10_challenge_quote_links.csv", "relationships")
    for _, r in cq_links.iterrows():
        ch_id = r.get("challenge_id")
        info_id = r.get("information_id")
        if not pd.isna(ch_id) and not pd.isna(info_id):
            ch_gid = f"challenge_{int(ch_id)}"
            info_gid = f"information_{int(info_id)}"
            if ch_gid in valid_ids and info_gid in valid_ids:
                add_edge(rows, ch_gid, info_gid, "challenge_supported_by_quote", "interpretive",
                         "listening", False, "challenge-quote link from p10 relink")

    # --- Agent → Agent interconnections ---
    for _, r in load_table("agents.csv").iterrows():
        agid = f"agent_{r['id']}"
        if agid not in valid_ids:
            continue
        inter_json = r.get("interconnections_json", "")
        if pd.isna(inter_json) or not inter_json.strip():
            continue
        try:
            conns = json.loads(inter_json) if isinstance(inter_json, str) else inter_json
        except Exception:
            continue
        for conn in conns if isinstance(conns, list) else []:
            other_id = conn.get("agent_id") or conn.get("id") or conn.get("interconnected_agent")
            if other_id:
                other_gid = f"agent_{other_id}"
                if other_gid in valid_ids and other_gid != agid:
                    add_edge(rows, agid, other_gid, "declared_interconnection", "declared_relational",
                             "mapping", False, "agent interconnection")

    df = pd.DataFrame(rows).drop_duplicates(subset=["source_global_id", "target_global_id", "edge_type"])
    return df


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Building graph tables for Platform {PLATFORM_ID} (Chile COPOLAD)")
    print(f"  Output: {ANALYTICS_DIR}")

    nodes = build_nodes()
    nodes = nodes.drop_duplicates(subset="global_id").reset_index(drop=True)
    print(f"  Nodes: {len(nodes)}")

    edges = build_edges(nodes)
    print(f"  Edges: {len(edges)}")
    print(f"  Edge types: {edges['edge_type'].value_counts().to_dict()}")

    # Merge with existing if present
    if NODES_CSV.exists():
        old_nodes = pd.read_csv(NODES_CSV)
        old_gids = set(old_nodes["global_id"].astype(str))
        new_nodes = nodes[~nodes["global_id"].astype(str).isin(old_gids)]
        nodes = pd.concat([old_nodes, new_nodes], ignore_index=True)
        print(f"  Merged nodes: {len(nodes)} ({len(new_nodes)} new)")

    if EDGES_CSV.exists():
        old_edges = pd.read_csv(EDGES_CSV)
        old_edges_clean = old_edges.drop(columns=["edge_id"], errors="ignore")
        old_keys = set(zip(old_edges_clean["source_global_id"].astype(str),
                           old_edges_clean["target_global_id"].astype(str),
                           old_edges_clean["edge_type"].astype(str)))
        edges_clean = edges.drop(columns=["edge_id"], errors="ignore")
        new_edges = edges_clean[~edges_clean.apply(lambda r: (str(r["source_global_id"]), str(r["target_global_id"]), str(r["edge_type"])), axis=1).isin(old_keys)]
        edges = pd.concat([old_edges_clean, new_edges], ignore_index=True)

    # Add edge_id at the end (after all merges)
    edges = edges.reset_index(drop=True)
    if "edge_id" in edges.columns:
        edges = edges.drop(columns=["edge_id"])
    edges.insert(0, "edge_id", [f"e{i+1}" for i in range(len(edges))])

    nodes.to_csv(NODES_CSV, index=False)
    edges.to_csv(EDGES_CSV, index=False)
    print(f"\nDone: {len(nodes)} nodes, {len(edges)} edges")
    print(f"  {NODES_CSV}")
    print(f"  {EDGES_CSV}")


if __name__ == "__main__":
    main()

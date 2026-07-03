import pandas as pd
import networkx as nx
import os
import re

PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
PROJECT_NAME = os.environ.get("KTOOL_PROJECT_NAME", "ALC")
DATA_DIR = f"data/processed/{PLATFORM_ID}/{OUTPUT_SUBDIR}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return slug.strip("_") or "project"


def export_graph_files(graph: nx.MultiDiGraph) -> tuple[str, str]:
    """Write both a project-aware and a compatibility GEXF output."""
    project_slug = slugify(PROJECT_NAME)
    named_gexf = (
        f"{project_slug}_platform_{PLATFORM_ID}_ecosystem_graph.gexf"
    )
    named_path = os.path.join(DATA_DIR, named_gexf)
    legacy_path = os.path.join(DATA_DIR, "ecosystem_graph.gexf")

    graph.graph["project_name"] = PROJECT_NAME
    graph.graph["platform_id"] = str(PLATFORM_ID)
    graph.graph["output_subdir"] = OUTPUT_SUBDIR
    graph.graph["graph_label"] = f"{PROJECT_NAME} ecosystem platform {PLATFORM_ID}"

    nx.write_gexf(graph, named_path)
    nx.write_gexf(graph, legacy_path)
    return named_path, legacy_path


def load_safe_csv(filename, subfolder=""):
    """Loads a CSV cleanly, checking both the flat root directory and subfolder layouts."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 10:
        filepath = os.path.join(DATA_DIR, subfolder, filename)
        
    if os.path.exists(filepath) and os.path.getsize(filepath) >= 10:
        try:
            df = pd.read_csv(filepath)
            if not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def load_any_csv(*filenames, subfolder=""):
    for filename in filenames:
        frame = load_safe_csv(filename, subfolder)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def clean_attr(value):
    if pd.isna(value):
        return ""
    return value


def clean_graph_attrs(row, columns, reserved=()):
    """Prepare attributes for GEXF without clobbering reserved XML ids."""
    attrs = {}
    reserved_set = set(reserved)
    for col in columns:
        if col in reserved_set:
            continue
        value = clean_attr(row[col])
        if value == "":
            continue
        if col == "id":
            attrs["source_record_id"] = value
        elif col in {"source", "target"}:
            attrs[f"semantic_{col}"] = value
        elif col == "weight":
            try:
                attrs[col] = float(value)
            except (TypeError, ValueError):
                attrs[col] = 1.0
        else:
            attrs[col] = value
    return attrs


def build_graph_from_normalized_tables():
    nodes_df = load_any_csv("nodes.csv", subfolder="analytics")
    edges_df = load_any_csv("edges.csv", subfolder="analytics")
    if nodes_df.empty or edges_df.empty:
        return None

    required_node_cols = {"global_id", "node_type", "label"}
    required_edge_cols = {"source_global_id", "target_global_id", "edge_type"}
    if not required_node_cols.issubset(nodes_df.columns) or not required_edge_cols.issubset(edges_df.columns):
        return None

    G = nx.MultiDiGraph()

    for _, row in nodes_df.iterrows():
        node_id = str(row["global_id"])
        if not node_id.strip():
            continue
        attrs = clean_graph_attrs(row, nodes_df.columns, reserved={"global_id"})
        G.add_node(node_id, **attrs)

    skipped_edges = 0
    for _, row in edges_df.iterrows():
        source = str(row["source_global_id"])
        target = str(row["target_global_id"])
        if source not in G or target not in G:
            skipped_edges += 1
            continue

        attrs = clean_graph_attrs(
            row,
            edges_df.columns,
            reserved={"source_global_id", "target_global_id"},
        )
        key = attrs.get("edge_id") or None
        G.add_edge(source, target, key=key, **attrs)

    print("Loaded normalized graph tables.")
    print(f"Nodes loaded from nodes.csv: {G.number_of_nodes()}")
    print(f"Edges loaded from edges.csv: {G.number_of_edges()}")
    if skipped_edges:
        print(f"Skipped normalized edges with missing endpoints: {skipped_edges}")
    return G


def build_graph():
    normalized_graph = build_graph_from_normalized_tables()
    if normalized_graph is not None:
        named_output_gexf, compatibility_output_gexf = export_graph_files(normalized_graph)
        print("========================================================================")
        print("GRAPH SYSTEM COMPILATION COMPLETE")
        print(f"Nodes Injected: {normalized_graph.number_of_nodes()} | Relationships Formed: {normalized_graph.number_of_edges()}")
        print(f"Primary output saved: {named_output_gexf}")
        print(f"Compatibility output saved: {compatibility_output_gexf}")
        print("========================================================================")
        return

    print("Normalized nodes.csv/edges.csv not found; falling back to legacy CSV stitching.")
    G = nx.MultiDiGraph()
    
    # --------------------------------------------------------------------------
    # 1. LOAD DATA TABLES
    # --------------------------------------------------------------------------
    df_agents = load_safe_csv("agents.csv", "entities")
    df_initiatives = load_safe_csv("initiatives_unified.csv", "analytics")
    df_perceptions = load_safe_csv("perceptions.csv", "entities")
    df_challenges = load_any_csv("challenges.csv", "challenges_opportunities.csv", subfolder="entities")
    df_patterns = load_safe_csv("patterns.csv", "entities")
    df_values = load_safe_csv("values.csv", "entities")
    df_channels = load_safe_csv("channels.csv", "entities")
    df_informations = load_safe_csv("informations.csv", "entities")
    df_sessions = load_safe_csv("sessions.csv", "entities")

    print(f"Loaded {len(df_agents)} agents and {len(df_initiatives)} unified initiatives matching your exact schema.")

    # --------------------------------------------------------------------------
    # 2. INGEST NODES (MATCHING YOUR EXACT HEADERS)
    # --------------------------------------------------------------------------
    print("Ingesting metadata attributes into graph structure...")

    # Agents
    if not df_agents.empty:
        id_col = 'agent_id' if 'agent_id' in df_agents.columns else 'id'
        for _, row in df_agents.iterrows():
            if id_col not in row or pd.isna(row[id_col]): continue
            G.add_node(
                f"agent_{row[id_col]}", label=row.get('name', 'Unnamed Agent'), node_type='agent',
                agent_type=str(row.get('type', 'Unknown')),
                investment=str(row.get('investment', 'None')),
                description=str(row.get('description', ''))[:400],
                people_involved=str(row.get('people_involved', '')),
                contact=str(row.get('contact', '')),
            )

    # Initiatives (Projects, Pilots, Prototypes) aligned with your columns
    if not df_initiatives.empty:
        for _, row in df_initiatives.iterrows():
            if 'initiative_global_id' not in row or pd.isna(row['initiative_global_id']): 
                continue
            
            node_id = str(row['initiative_global_id'])
            G.add_node(
                node_id, 
                label=row.get('title', 'Unnamed Initiative'), 
                node_type='initiative',
                init_type=str(row.get('initiative_type', 'initiative')), 
                sector=str(row.get('sector', 'Unknown')),
                impact_level=str(row.get('impact_level', 'Unknown')), 
                disruption_level=str(row.get('disruption_level', 'Unknown')),
                status=str(row.get('status', 'Unknown')),
                description=str(row.get('description', 'No description loaded'))[:400]
            )

    # Core qualitative layers
    if not df_perceptions.empty:
        for _, row in df_perceptions.iterrows(): G.add_node(f"perception_{row['id']}", label=row.get('name', 'Unnamed Perception'), node_type='perception')
    if not df_challenges.empty:
        for _, row in df_challenges.iterrows(): G.add_node(f"challenge_{row['id']}", label=row.get('name', 'Unnamed Challenge'), node_type='challenge')
    if not df_patterns.empty:
        for _, row in df_patterns.iterrows(): G.add_node(f"pattern_{row['id']}", label=row.get('name', 'Unnamed Pattern'), node_type='pattern')
    if not df_values.empty:
        for _, row in df_values.iterrows(): G.add_node(f"value_{row['id']}", label=row.get('name', 'Unnamed Value'), node_type='value')
    if not df_channels.empty:
        for _, row in df_channels.iterrows(): G.add_node(f"channel_{row['id']}", label=row.get('name', 'Unnamed Channel'), node_type='channel')
    if not df_informations.empty:
        for _, row in df_informations.iterrows(): G.add_node(f"info_{row['id']}", label=f"Quote Info {row['id']}", node_type='information', quote_text=str(row.get('quote', ''))[:150])
    if not df_sessions.empty:
        for _, row in df_sessions.iterrows(): G.add_node(f"session_{row['id']}", label=row.get('name', 'Unnamed Session'), node_type='session')

    # --------------------------------------------------------------------------
    # 3. STITCH RELATIONSHIP EDGES
    # --------------------------------------------------------------------------
    print("Stitching system edge networks...")

    def bind_edges(filename, subfolder, source_prefix, target_prefix, source_col, target_col, rel_label):
        df_edge = load_safe_csv(filename, subfolder)
        if df_edge.empty: return
        
        # Resolve column naming variants gracefully
        actual_src = source_col if source_col in df_edge.columns else ('id' if source_col == 'agent_id' else ('agent_id' if source_col == 'id' else source_col))
        actual_tgt = target_col if target_col in df_edge.columns else ('id' if target_col == 'agent_id' else ('agent_id' if target_col == 'id' else target_col))
        if source_col == 'initiative_global_id' and 'global_id' in df_edge.columns: actual_src = 'global_id'
        if target_col == 'initiative_global_id' and 'global_id' in df_edge.columns: actual_tgt = 'global_id'

        if actual_src not in df_edge.columns or actual_tgt not in df_edge.columns: return

        for _, row in df_edge.iterrows():
            if pd.isna(row[actual_src]) or pd.isna(row[actual_tgt]): continue
            s = f"{source_prefix}{row[actual_src]}" if source_prefix else str(row[actual_src])
            t = f"{target_prefix}{row[actual_tgt]}" if target_prefix else str(row[actual_tgt])
            if s in G and t in G:
                G.add_edge(s, t, rel_type=rel_label)

    # Bind active system lines
    bind_edges("agent_initiative_links.csv", "relationships", "agent_", "", "agent_id", "initiative_global_id", "collaborates")
    bind_edges("initiative_lead_agent_links.csv", "relationships", "", "agent_", "initiative_global_id", "agent_id", "lead_agent")
    bind_edges("initiative_partner_links.csv", "relationships", "", "", "initiative_global_id", "partner_sector", "partner")
    bind_edges("initiative_thematic_area_links.csv", "relationships", "", "theme_", "initiative_global_id", "thematic_area_id", "tagged_theme")
    bind_edges("initiative_perception_links.csv", "relationships", "", "perception_", "initiative_global_id", "perception_id", "addresses")
    bind_edges("channel_information_links.csv", "relationships", "channel_", "info_", "channel_id", "information_id", "contains")
    bind_edges("information_pattern_links.csv", "relationships", "info_", "pattern_", "information_id", "pattern_id", "clusters_into")
    bind_edges("information_value_links.csv", "relationships", "info_", "value_", "information_id", "value_id", "expresses_value")
    bind_edges("pattern_perception_links.csv", "relationships", "pattern_", "perception_", "pattern_id", "perception_id", "feeds_narrative")
    bind_edges("perception_challenge_links.csv", "relationships", "perception_", "challenge_", "perception_id", "challenge_id", "diagnoses")
    bind_edges("value_perception_links.csv", "relationships", "value_", "perception_", "value_id", "perception_id", "frames")
    bind_edges("information_pattern_connections.csv", "relationships", "info_", "pattern_", "information_id", "pattern_id", "pattern_connection")
    
    # Interconnections layer
    df_interconn = load_any_csv("initiative_initiative_links.csv", "link_initiative_interconnections.csv", subfolder="relationships")
    if not df_interconn.empty:
        for _, row in df_interconn.iterrows():
            s = str(row.get('source_initiative_global_id', row.get('source_prototype_id')))
            t = str(row.get('target_global_id', row.get('target_initiative_global_id', row.get('target_prototype_id'))))
            if s in G and t in G:
                G.add_edge(s, t, rel_type="interconnected")
            elif s in G and t.startswith("agent_") and t in G:
                G.add_edge(s, t, rel_type="interconnected")
            elif s in G and t.startswith("perception_") and t in G:
                G.add_edge(s, t, rel_type="interconnected")

    # --------------------------------------------------------------------------
    # 4. EXPORT COMPREHENSIVE GEXF
    # --------------------------------------------------------------------------
    named_output_gexf, compatibility_output_gexf = export_graph_files(G)
    print("========================================================================")
    print(f"🎉 GRAPH SYSTEM COMPILATION COMPLETE!")
    print(f"Nodes Injected: {G.number_of_nodes()} | Relationships Formed: {G.number_of_edges()}")
    print(f"Primary output saved: {named_output_gexf}")
    print(f"Compatibility output saved: {compatibility_output_gexf}")
    print("========================================================================")

if __name__ == "__main__":
    build_graph()

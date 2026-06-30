import json
import os
import networkx as nx
import pandas as pd

def compile_ireland_network():
    platform_id = "173"
    raw_dir = f"data/{platform_id}"
    output_dir = f"data/processed/{platform_id}"
    os.makedirs(output_dir, exist_ok=True)

    project_file = f"{raw_dir}/projects.json"
    
    print("========================================================================")
    print(f"🧠 RUNNING TOPOLOGICAL MASTER COMPILER (PLATFORM {platform_id} - IRELAND)")
    print("========================================================================")

    if not os.path.exists(project_file):
        print(f"❌ Target project data missing at {project_file}. Please run extraction first.")
        return

    with open(project_file, "r", encoding="utf-8") as f:
        raw_payload = json.load(f)
        
    # Strapi lists records inside a 'data' array
    projects = raw_payload.get("data", [])
    print(f"📦 Data Assets Loaded: {len(projects)} Core Projects")

    # Initialize NetworkX Graph
    G = nx.Graph()
    node_types = {}

    # 1. GRAPH GENERATION LAYER (Building the Multipartite Architecture)
    for p in projects:
        p_id = p.get("id")
        attrs = p.get("attributes", {})
        p_name = attrs.get("name")
        
        if not p_name:
            continue
            
        p_label = f"Proj_{p_id}: {p_name}"
        G.add_node(p_label, type="project")
        node_types[p_label] = "project"

        # A. Map Perceptions (Macro Pillars)
        perceptions = attrs.get("perceptions", {}).get("data", [])
        for per in perceptions:
            per_attrs = per.get("attributes", {})
            per_name = per_attrs.get("name") or f"Perception #{per.get('id')}"
            per_label = f"Macro: {per_name}"
            
            G.add_node(per_label, type="perception")
            node_types[per_label] = "perception"
            G.add_edge(p_label, per_label, relation="assigned_to")

        # B. Map Agents (Lead & Collaborators)
        # Check explicit lead agent field or nested data structure
        lead_agent = attrs.get("lead_agent")
        agents_list = attrs.get("agents", {}).get("data", [])
        
        all_agents = []
        if lead_agent:
            all_agents.append(str(lead_agent))
        for ag in agents_list:
            ag_attrs = ag.get("attributes", {})
            ag_name = ag_attrs.get("name") or ag_attrs.get("username")
            if ag_name:
                all_agents.append(ag_name)
                
        for agent_name in set(all_agents):
            agent_label = f"Agent: {agent_name}"
            G.add_node(agent_label, type="agent")
            node_types[agent_label] = "agent"
            G.add_edge(p_label, agent_label, relation="involved_in")

        # C. Map Partner Sectors (Civil Society, Public Admin, etc.)
        partners = attrs.get("partners")
        if partners:
            if isinstance(partners, str):
                partners = [partners]
            elif isinstance(partners, dict) and "data" in partners:
                partners = [item.get("attributes", {}).get("name") for item in partners.get("data", []) if item.get("attributes", {}).get("name")]
                
            for partner in partners:
                if partner:
                    partner_label = f"Sector: {partner}"
                    G.add_node(partner_label, type="partner")
                    node_types[partner_label] = "partner"
                    G.add_edge(p_label, partner_label, relation="aligned_with")

        # D. Map Thematic Areas
        themes = attrs.get("thematic_areas")
        if themes:
            if isinstance(themes, str):
                themes = [themes]
            elif isinstance(themes, dict) and "data" in themes:
                themes = [item.get("attributes", {}).get("name") for item in themes.get("data", []) if item.get("attributes", {}).get("name")]
                
            for theme in themes:
                if theme:
                    theme_label = f"Theme: {theme}"
                    G.add_node(theme_label, type="theme")
                    node_types[theme_label] = "theme"
                    G.add_edge(p_label, theme_label, relation="tagged_with")

    nx.set_node_attributes(G, node_types, "type")
    print(f"🕸️ Network Topology Compiled: {G.number_of_nodes()} total entities | {G.number_of_edges()} structural edges")

    if G.number_of_nodes() == 0:
        print("⚠️ Graph is empty. Verify that the project records match the expected attributes schema.")
        return

    # ------------------------------------------------------------------
    # 2. RUNNING GRAPH THEORETIC ANALYSIS
    # ------------------------------------------------------------------
    print("\n📊 EXECUTING SYSTEM DIAGNOSTICS LAYER")
    print("========================================================================")

    # MACRO ANALYSIS: Global Fragmentations & Modularity
    from networkx.algorithms import community
    communities = list(community.greedy_modularity_communities(G))
    modularity_score = community.modularity(G, communities)
    
    print(f"📈 MACRO: Modularity Score = {modularity_score:.4f} ({len(communities)} isolated structural sub-worlds detected)")

    # MESO ANALYSIS: Structural Tension via Betweenness Centrality
    betweenness = nx.betweenness_centrality(G)
    sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    
    meso_records = []
    print("\n🔥 MESO: Top 5 High-Tension Structural Brokerage Points (System Pipelines):")
    print("-" * 72)
    count = 0
    for node, score in sorted_betweenness:
        n_type = node_types.get(node, "unknown")
        if n_type in ["project", "agent"] and count < 5:
            print(f"  ↳ Node: {node:<50} | Centrality Vector: {score:.4f}")
            count += 1
        meso_records.append({"node": node, "type": n_type, "betweenness_centrality": score})
    
    # MICRO ANALYSIS: Plasticity & Boundary Spanning Cases
    micro_records = []
    print("\n🎯 MICRO: Boundary-Spanning Nodes (Proving Narrative Elasticity/Change Possible):")
    print("-" * 72)
    boundary_count = 0
    for node in G.nodes():
        if node_types.get(node) == "project":
            neighbors = list(G.neighbors(node))
            linked_macro_pillars = [n for n in neighbors if node_types.get(n) == "perception"]
            linked_themes = [n for n in neighbors if node_types.get(n) == "theme"]
            
            participation_score = len(linked_macro_pillars)
            micro_records.append({
                "project": node,
                "macro_pillar_count": participation_score,
                "themes_count": len(linked_themes),
                "is_boundary_spanner": participation_score > 1
            })
            
            if participation_score > 1 and boundary_count < 5:
                print(f"  ↳ {node:<50} | Anchors {participation_score} distinct Macro Pillars simultaneously.")
                boundary_count += 1

    # ------------------------------------------------------------------
    # 3. EXPORT MATRICES TO CSV
    # ------------------------------------------------------------------
    pd.DataFrame(meso_records).to_csv(f"{output_dir}/meso_structural_tensions.csv", index=False)
    pd.DataFrame(micro_records).to_csv(f"{output_dir}/micro_project_plasticity.csv", index=False)
    
    # Export clean edge list file for immediate ingestion into Gephi / Kumu
    edges_df = nx.to_pandas_edgelist(G)
    edges_df.to_csv(f"{output_dir}/network_edge_list.csv", index=False)
    
    print("\n💾 STRATEGIC DIAGNOSTICS ARCHIVED")
    print("========================================================================")
    print(f" • Full Gephi/Kumu Edge List Matrix saved to : {output_dir}/network_edge_list.csv")
    print(f" • Meso Tension Data Frame saved to          : {output_dir}/meso_structural_tensions.csv")
    print(f" • Micro Plasticity Data Frame saved to      : {output_dir}/micro_project_plasticity.csv")

if __name__ == "__main__":
    compile_ireland_network()
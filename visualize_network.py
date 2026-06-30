import json
import os
import networkx as nx
import matplotlib.pyplot as plt

def generate_visual_map():
    platform_id = "173"
    raw_dir = f"data/{platform_id}"
    output_dir = f"data/processed/{platform_id}"
    os.makedirs(output_dir, exist_ok=True)

    project_file = f"{raw_dir}/projects.json"
    
    print("========================================================================")
    print(f"🎨 GENERATING VISUAL TOPOLOGY MAP (PLATFORM {platform_id})")
    print("========================================================================")

    # 1. Rebuild the Graph
    with open(project_file, "r", encoding="utf-8") as f:
        projects = json.load(f).get("data", [])

    G = nx.Graph()
    node_types = {}

    for p in projects:
        attrs = p.get("attributes", {})
        p_name = attrs.get("name")
        if not p_name: continue
            
        p_label = f"Proj: {p_name[:15]}..." # Truncate for visual clarity
        G.add_node(p_label, type="project")
        node_types[p_label] = "project"

        # Perceptions
        for per in attrs.get("perceptions", {}).get("data", []):
            per_name = per.get("attributes", {}).get("name", "Unknown")
            per_label = f"Macro: {per_name}"
            G.add_node(per_label, type="perception")
            node_types[per_label] = "perception"
            G.add_edge(p_label, per_label)

        # Agents
        lead_agent = attrs.get("lead_agent")
        agents_list = attrs.get("agents", {}).get("data", [])
        all_agents = [str(lead_agent)] if lead_agent else []
        all_agents.extend([ag.get("attributes", {}).get("name") for ag in agents_list if ag.get("attributes", {}).get("name")])
        
        for agent_name in set(all_agents):
            agent_label = f"Agent: {agent_name[:15]}"
            G.add_node(agent_label, type="agent")
            node_types[agent_label] = "agent"
            G.add_edge(p_label, agent_label)

    nx.set_node_attributes(G, node_types, "type")
    
    if G.number_of_nodes() == 0:
        print("⚠️ Graph is empty.")
        return

    # 2. Configure Visual Aesthetics
    # Color mapping dictionary
    color_map = {
        "project": "#1f77b4",    # Blue
        "agent": "#d62728",      # Red
        "perception": "#2ca02c", # Green
        "partner": "#9467bd",    # Purple
        "theme": "#ff7f0e"       # Orange
    }

    # Assign colors and sizes based on network metrics
    node_colors = [color_map.get(G.nodes[n].get("type", "project"), "#333333") for n in G.nodes()]
    
    # Scale node size by its number of connections (Degree)
    degrees = dict(G.degree())
    node_sizes = [v * 50 + 100 for v in degrees.values()] 

    # 3. Draw the Network
    plt.figure(figsize=(16, 12), facecolor="#f8f9fa")
    plt.title(f"Irish Sensemaking Network (Platform {platform_id})\nNodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}", fontsize=18, pad=20)
    
    # Using a Spring Layout to push highly connected nodes to the center and isolated nodes to the edges
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

    # Draw elements
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color="#999999")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors="white", linewidths=1.5)
    
    # Only label the most critical "Bridge" nodes (Degree > 3) to prevent text clutter
    labels = {n: n for n in G.nodes() if degrees[n] > 3 or G.nodes[n].get("type") == "perception"}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold", font_color="#333333")

    # Legend
    markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='', markersize=10) for color in color_map.values()]
    plt.legend(markers, color_map.keys(), numpoints=1, loc="upper right", title="Node Types", fontsize=12)

    plt.axis("off")
    plt.tight_layout()

    # 4. Export Image
    output_img = f"{output_dir}/network_map_high_res.png"
    plt.savefig(output_img, dpi=300, bbox_inches="tight")
    print(f"✅ High-Resolution Network Map saved to: {output_img}")

if __name__ == "__main__":
    generate_visual_map()
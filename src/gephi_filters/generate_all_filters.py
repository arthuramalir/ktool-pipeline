"""Generate filtered GEXF + GraphML files for Gephi.
One file per analytical view. Run from project root:
  python src/gephi_filters/generate_all_filters.py
"""

import os, re, sys
from pathlib import Path
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from graph_utils import build_graph, simple_graph, ANALYSIS_DIR, DATA_DIR, PLATFORM_ID, OUTPUT_SUBDIR

FILTERS_DIR = DATA_DIR / "analysis" / "gephi_filters"
FILTERS_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_NAME = os.environ.get("KTOOL_PROJECT_NAME", "ALC")

def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return slug.strip("_") or "project"

def label_file(name):
    prefix = f"{slugify(PROJECT_NAME)}_platform_{PLATFORM_ID}_"
    return FILTERS_DIR / f"{prefix}{name}"

VIEWS = []

def register(name, desc, G, default_attrs=True):
    """Export a graph as GEXF + GraphML and register it."""
    if G.number_of_nodes() == 0:
        print(f"  SKIP {name}: 0 nodes")
        return
    gexf = label_file(f"{name}.gexf")
    graphml = label_file(f"{name}.graphml")
    nx.write_gexf(G, str(gexf))
    nx.write_graphml(G, str(graphml))
    VIEWS.append({
        "name": name,
        "description": desc,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "gexf": str(gexf.relative_to(DATA_DIR)),
        "graphml": str(graphml.relative_to(DATA_DIR)),
    })
    print(f"  OK  {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# =========================================================================
# 1. BASE GRAPHS
# =========================================================================
print("\n=== 1. BASE GRAPHS ===")

G_all = build_graph(directed=False)
G_simple = simple_graph(G_all)

register("full_graph", "Full graph — all 783 edges (declared + semantic + listening + interpretive)", G_simple)

# Source-data only (exclude qualitative_narrative)
G_source = build_graph(directed=False)
G_source_simple = simple_graph(G_source)
keep = []
for u, v, d in G_source_simple.edges(data=True):
    fam = d.get("edge_family", "")
    if fam != "qualitative_narrative":
        keep.append((u, v, d))
G_source_only = nx.Graph()
G_source_only.add_nodes_from(G_source_simple.nodes(data=True))
G_source_only.add_edges_from(keep)
register("source_data_only", "Source-data edges only (540 declared + 71 listening + 50 interpretive) — no semantic edges", G_source_only)

# Semantic edges only
G_sem = nx.Graph()
sem_edges = []
for u, v, d in G_simple.edges(data=True):
    if d.get("edge_family", "") == "qualitative_narrative":
        sem_edges.append((u, v, d))
sem_nodes = set()
for u, v, d in sem_edges:
    sem_nodes.add(u); sem_nodes.add(v)
for n in sem_nodes:
    attrs = G_simple.nodes[n]
    G_sem.add_node(n, **attrs)
G_sem.add_edges_from(sem_edges)
register("semantic_only", "Semantic edges only (120 manually-analysed quote-to-quote edges)", G_sem)

# =========================================================================
# 2. FILTER BY EDGE TYPE
# =========================================================================
print("\n=== 2. BY EDGE TYPE ===")

for etype in ["similarity", "contradiction", "causality", "sequence"]:
    G_filt = nx.Graph()
    filt_edges = [(u, v, d) for u, v, d in G_simple.edges(data=True)
                  if d.get("edge_type", "") == etype]
    fnodes = set()
    for u, v, d in filt_edges:
        fnodes.add(u); fnodes.add(v)
    for n in fnodes:
        attrs = G_simple.nodes[n]
        G_filt.add_node(n, **attrs)
    G_filt.add_edges_from(filt_edges)
    register(f"edge_type_{etype}",
             f"Only {etype} edges ({len(filt_edges)} edges)", G_filt)

# =========================================================================
# 3. LARGEST COMPONENT
# =========================================================================
print("\n=== 3. LARGEST COMPONENT ===")

comp = list(nx.connected_components(G_simple))
if comp:
    largest = max(comp, key=len)
    G_largest = G_simple.subgraph(largest).copy()
    register("largest_component",
             f"Largest connected component — {len(largest)} nodes ({len(largest)/G_simple.number_of_nodes()*100:.1f}% of graph)",
             G_largest)

    # Largest component, source-data only
    keep_edges = [(u, v, d) for u, v, d in G_largest.edges(data=True)
                  if d.get("edge_family", "") != "qualitative_narrative"]
    G_largest_source = nx.Graph()
    G_largest_source.add_nodes_from(G_largest.nodes(data=True))
    G_largest_source.add_edges_from(keep_edges)
    register("largest_component_source_only",
             "Largest component — source-data edges only (semantic hidden)", G_largest_source)

    # Largest component, semantic only
    sem_edges_lc = [(u, v, d) for u, v, d in G_largest.edges(data=True)
                    if d.get("edge_family", "") == "qualitative_narrative"]
    G_largest_sem = nx.Graph()
    G_largest_sem.add_nodes_from(G_largest.nodes(data=True))
    G_largest_sem.add_edges_from(sem_edges_lc)
    register("largest_component_semantic_only",
             "Largest component — qualitative_narrative edges only", G_largest_sem)

# =========================================================================
# 4. FILTER BY NODE TYPE
# =========================================================================
print("\n=== 4. BY NODE TYPE ===")

for ntype in ["agent", "project", "information", "channel", "challenge", "perception"]:
    G_nt = nx.Graph()
    nt_edges = []
    nt_nodes = set()
    for n, attrs in G_simple.nodes(data=True):
        if attrs.get("node_type", "") == ntype:
            nt_nodes.add(n)
    for u, v, d in G_simple.edges(data=True):
        if u in nt_nodes or v in nt_nodes:
            nt_edges.append((u, v, d))
    for n in nt_nodes:
        G_nt.add_node(n, **G_simple.nodes[n])
    G_nt.add_edges_from(nt_edges)
    register(f"node_type_{ntype}",
             f"Subgraph centred on {ntype} nodes ({len(nt_nodes)} {ntype}s)", G_nt)

# =========================================================================
# 5. COMBINATION VIEWS
# =========================================================================
print("\n=== 5. COMBINATION VIEWS ===")

# Agents with semantic edges only
G_agent_sem = nx.Graph()
agent_sem_edges = []
agent_sem_nodes = set()
for n, attrs in G_simple.nodes(data=True):
    if attrs.get("node_type", "") == "agent":
        agent_sem_nodes.add(n)
for u, v, d in G_simple.edges(data=True):
    if d.get("edge_family", "") == "qualitative_narrative":
        agent_sem_edges.append((u, v, d))
        if u in agent_sem_nodes: agent_sem_nodes.add(u)
        if v in agent_sem_nodes: agent_sem_nodes.add(v)
for n in agent_sem_nodes:
    G_agent_sem.add_node(n, **G_simple.nodes[n])
G_agent_sem.add_edges_from(agent_sem_edges)
register("agents_via_semantic",
         "Agents connected via semantic edges — shows which agents share narrative framings", G_agent_sem)

# Information nodes only (quote network)
G_info = nx.Graph()
info_edges = []
info_nodes = set()
for n, attrs in G_simple.nodes(data=True):
    if attrs.get("node_type", "") == "information":
        info_nodes.add(n)
for u, v, d in G_simple.edges(data=True):
    if u in info_nodes or v in info_nodes:
        info_edges.append((u, v, d))
for n in info_nodes:
    G_info.add_node(n, **G_simple.nodes[n])
G_info.add_edges_from(info_edges)
register("information_network",
         "Information (quote) nodes and all their connections — the narrative layer", G_info)

# =========================================================================
# 6. SUMMARY TABLE
# =========================================================================
print("\n=== SUMMARY ===")
summary_path = label_file("filter_inventory.csv")
pd.DataFrame(VIEWS).to_csv(summary_path, index=False)
print(f"\n{len(VIEWS)} views exported to {FILTERS_DIR}")
print(f"Inventory: {summary_path}")
for v in VIEWS:
    print(f"  {v['name']:45s}  {v['nodes']:4d} nodes  {v['edges']:4d} edges")

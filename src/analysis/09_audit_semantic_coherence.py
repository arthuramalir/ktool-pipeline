from __future__ import annotations
import pandas as pd
from graph_utils import load_nodes_edges

def audit_semantic_coherence() -> None:
    print("Initializing Semantic Coherence Audit...")
    
    # 1. Load the master nodes and edges
    nodes, edges = load_nodes_edges()
    if nodes.empty or edges.empty:
        print("[ERROR] Base graph files are missing or empty.")
        return
        
    # Build a fast dictionary lookup for descriptions and labels
    node_text_map = nodes.set_index("global_id")[["label", "node_type", "description"]].to_dict(orient="index")
    
    # 2. Isolate only the NLP Semantic Edges from the master edge table
    df_nlp_edges = edges[edges["edge_family"] == "interpretive_nlp"].copy()
    if df_nlp_edges.empty:
        print("[WARNING] Zero semantic edges found in the master edges.csv file.")
        return

    # Convert weights to float and sort descending to see strongest matches first
    df_nlp_edges["weight"] = df_nlp_edges["weight"].astype(float)
    df_nlp_edges = df_nlp_edges.sort_values(by="weight", ascending=False)
    
    print(f"\n=======================================================")
    print(f"   SOCIO-SEMANTIC COHERENCE REPORT ({len(df_nlp_edges)} total links found)")
    print(f"=======================================================\n")
    
    # 3. Print the top 5 strongest semantic bridges
    print("--- TOP 5 STRONGEST SEMANTIC CONNECTIONS ---")
    top_5 = df_nlp_edges.head(5)
    for _, row in top_5.iterrows():
        src, tgt, weight = row["source_global_id"], row["target_global_id"], row["weight"]
        src_meta = node_text_map.get(src, {"label": "Unknown", "node_type": "Unknown", "description": ""})
        tgt_meta = node_text_map.get(tgt, {"label": "Unknown", "node_type": "Unknown", "description": ""})
        
        print(f"🔗 Cosine Similarity: {weight:.4f}")
        print(f"   [Source] ({src_meta['node_type']}) {src_meta['label']}: \"{str(src_meta['description'])[:140]}...\"")
        print(f"   [Target] ({tgt_meta['node_type']}) {tgt_meta['label']}: \"{str(tgt_meta['description'])[:140]}...\"")
        print("-" * 50)

    print("\n")

    # 4. Print the bottom 5 weakest semantic bridges (right at your threshold boundary)
    print("--- BOTTOM 5 WEAKEST SEMANTIC CONNECTIONS (Boundary Cleanliness) ---")
    bottom_5 = df_nlp_edges.tail(5)
    for _, row in bottom_5.iterrows():
        src, tgt, weight = row["source_global_id"], row["target_global_id"], row["weight"]
        src_meta = node_text_map.get(src, {"label": "Unknown", "node_type": "Unknown", "description": ""})
        tgt_meta = node_text_map.get(tgt, {"label": "Unknown", "node_type": "Unknown", "description": ""})
        
        print(f"🔗 Cosine Similarity: {weight:.4f}")
        print(f"   [Source] ({src_meta['node_type']}) {src_meta['label']}: \"{str(src_meta['description'])[:140]}...\"")
        print(f"   [Target] ({tgt_meta['node_type']}) {tgt_meta['label']}: \"{str(tgt_meta['description'])[:140]}...\"")
        print("-" * 50)


if __name__ == "__main__":
    audit_semantic_coherence()
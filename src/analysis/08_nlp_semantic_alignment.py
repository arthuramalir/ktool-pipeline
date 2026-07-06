from __future__ import annotations
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from graph_utils import ANALYSIS_DIR, ANALYTICS_DIR, load_nodes_edges, write_frame

def run_listening_layer_nlp(similarity_threshold: float = 0.65) -> None:
    print("Initializing Semantic Listening Layer...")
    nodes, edges = load_nodes_edges()
    
    # Filter for nodes with text
    nodes_with_text = nodes[nodes["description"].fillna("").str.strip() != ""].copy()
    
    # 1. Define our two distinct layers
    LISTENING_TYPES = {"information", "perception", "value", "challenge", "quote"}
    OPERATIONAL_TYPES = {"project", "pilot", "prototype", "agent"}
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_text = nodes_with_text["description"].tolist()
    embeddings = model.encode(corpus_text, show_progress_bar=True)
    similarity_matrix = cosine_similarity(embeddings)

    # Also save per-node embeddings for GNN pipeline
    all_nodes = nodes.copy()
    all_nodes.set_index("global_id", inplace=True)
    embedding_dim = embeddings.shape[1]
    full_embedding_matrix = np.zeros((len(all_nodes), embedding_dim), dtype=np.float32)
    for idx, node_id in enumerate(nodes_with_text["global_id"].tolist()):
        if node_id in all_nodes.index:
            pos = all_nodes.index.get_loc(node_id)
            full_embedding_matrix[pos] = embeddings[idx]
    embed_df = pd.DataFrame({
        "global_id": all_nodes.index,
        "embedding": [full_embedding_matrix[i].tolist() for i in range(len(all_nodes))],
    })
    embed_out = ANALYSIS_DIR / "node_semantic_embeddings.parquet"
    embed_df.to_parquet(embed_out, index=False)
    print(f"Saved {len(embed_df)} node embeddings to {embed_out}")

    new_semantic_edges = []
    edge_counter = 1
    
    node_ids = nodes_with_text["global_id"].tolist()
    node_types = nodes_with_text["node_type"].tolist()
    
    # 2. Strict Bipartite Matching (Only match Listening to Operational)
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            score = similarity_matrix[i][j]
            
            if score >= similarity_threshold:
                type_i = str(node_types[i]).lower()
                type_j = str(node_types[j]).lower()
                
                # Check if one is a narrative/quote and the other is an operation/agent
                is_i_listening = type_i in LISTENING_TYPES
                is_j_listening = type_j in LISTENING_TYPES
                is_i_operational = type_i in OPERATIONAL_TYPES
                is_j_operational = type_j in OPERATIONAL_TYPES
                
                valid_match = (is_i_listening and is_j_operational) or (is_j_listening and is_i_operational)
                
                if valid_match:
                    new_semantic_edges.append({
                        "edge_id": f"narrative_link_{edge_counter}",
                        "source_global_id": node_ids[i],
                        "target_global_id": node_ids[j],
                        "edge_type": "narrative_to_operation_alignment",
                        "edge_family": "interpretive_nlp",
                        "directed": False,
                        "weight": round(float(score), 4),
                        "evidence_source": "Cross-layer cosine similarity"
                    })
                    edge_counter += 1

    df_semantic_edges = pd.DataFrame(new_semantic_edges)
    
    if not df_semantic_edges.empty:
        write_frame(df_semantic_edges, "nlp_semantic_edges.csv")
        print(f"Success! Anchored {len(df_semantic_edges)} raw narratives to actual projects/agents.")
        
        # Strip old NLP edges to prevent duplication, then append new ones
        clean_edges = edges[edges["edge_family"] != "interpretive_nlp"]
        combined_edges = pd.concat([clean_edges, df_semantic_edges], ignore_index=True)
        combined_edges.to_csv(ANALYTICS_DIR / "edges.csv", index=False)
    else:
        print("No cross-layer narrative matches found at this threshold.")

if __name__ == "__main__":
    # Notice the threshold is lowered slightly (0.65) because citizen quotes 
    # use different vocabulary than formal project descriptions.
    run_listening_layer_nlp(similarity_threshold=0.65)
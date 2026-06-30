from __future__ import annotations

from pathlib import Path
import pandas as pd

try:
    import spacy
    from spacy.matcher import Matcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from graph_utils import ANALYTICS_DIR, load_nodes_edges, write_frame


def compile_hypergraph_claims() -> None:
    print("🧠 Initializing Phase 2: Semantic Hypergraph Claims Compiler...")
    
    if not SPACY_AVAILABLE:
        print("[ERROR] Please run: pip install spacy && python -m spacy download en_core_web_sm")
        return

    # Load English language model
    nlp = spacy.load("en_core_web_sm")
    
    # 1. Ingest Master Ecosystem Nodes
    nodes, _ = load_nodes_edges()
    if nodes.empty:
        print("[ERROR] Nodes database is empty.")
        return

    # Create lookups for coreference and cross-linking resolution
    operational_nodes = nodes[nodes["node_type"].isin(["project", "pilot", "agent", "prototype"])]
    node_directory = operational_nodes.set_index("global_id")["label"].to_dict()
    
    # Isolate our narrative listening layers
    narrative_nodes = nodes[nodes["node_type"].isin(["information", "perception", "challenge", "value"])].copy()
    narrative_nodes = narrative_nodes[narrative_nodes["description"].fillna("").str.strip() != ""]

    print(f"Parsing {len(narrative_nodes)} qualitative text records into linguistic trees...")
    
    extracted_claims = []
    claim_idx = 1

    # 2. Linguistic Processing Loop
    for _, row in narrative_nodes.iterrows():
        text = str(row["description"])
        node_id = row["global_id"]
        node_type = row["node_type"]
        
        doc = nlp(text)
        
        # We parse the text sentence by sentence to find actionable impact mechanics
        for sent in doc.sents:
            root_verb = None
            subj = None
            obj = None
            prep_phrase = []

            # Extract subject, main verb, and object using dependency markers
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    root_verb = token.text
                elif token.dep_ in ("nsubj", "nsubjpass"):
                    # Extract the full noun phrase for the subject
                    subj = "".join([w.text_with_ws for w in token.subtree]).strip()
                elif token.dep_ in ("dobj", "pobj", "attr"):
                    obj = "".join([w.text_with_ws for w in token.subtree]).strip()
            
            # Extract trailing informational contexts (prepositions like 'in', 'through', 'thanks to')
            for token in sent:
                if token.dep_ == "prep":
                    p_phrase = "".join([w.text_with_ws for w in token.subtree]).strip()
                    prep_phrase.append(p_phrase)

            # If we found at least a basic Action dynamic, formulate a Graphbrain-style expression
            if root_verb and (subj or obj):
                subj_clean = subj or "someone"
                obj_clean = obj or "something"
                prep_str = f" ({' '.join(prep_phrase)})" if prep_phrase else ""
                
                # Format exactly like the Socsemics Hypergraph string representation:
                # (operator/verb subject object (prepositional_context))
                hypergraph_expr = f"({root_verb.lower()}/P {subj_clean.replace(' ', '_')}/C {obj_clean.replace(' ', '_')}/C{prep_str})"
                
                # 3. Coreference Resolution & Cross-Linking Engine
                # Scan to see if an official operational project/agent is named inside this text claim
                anchored_node_id = "unanchored_narrative"
                anchored_node_label = "Implicit Concept"
                
                for op_id, op_label in node_directory.items():
                    if str(op_label).lower() in text.lower():
                        anchored_node_id = op_id
                        anchored_node_label = str(op_label)
                        break

                extracted_claims.append({
                    "claim_id": f"claim_{claim_idx}",
                    "narrative_source_id": node_id,
                    "narrative_type": node_type,
                    "anchored_target_id": anchored_node_id,
                    "anchored_target_label": anchored_node_label,
                    "raw_sentence": sent.text.strip(),
                    "extracted_verb_operator": root_verb.lower(),
                    "hypergraph_expression": hypergraph_expr,
                    "impact_mechanic_summary": f"{subj_clean} -> {root_verb.upper()} -> {obj_clean}"
                })
                claim_idx += 1

    df_claims = pd.DataFrame(extracted_claims)
    
    # 4. Save out the structured Semantic Hypergraph Layer
    if not df_claims.empty:
        write_frame(df_claims, "socio_semantic_hypergraph_claims.csv")
        print(f"\n🚀 Success! Synthesized {len(df_claims)} Hypergraph Claims from the Listening Layer.")
        
        # Print a preview of the extracted structural assertions
        print("\n--- SAMPLE EXTRACTED HYPERGRAPH EXPRESSIONS ---")
        for _, c_row in df_claims.head(4).iterrows():
            print(f"\n📝 Text: \"{c_row['raw_sentence']}\"")
            print(f"   🤖 Hypergraph Expression: {c_row['hypergraph_expression']}")
            print(f"   🔗 Anchored Target: {c_row['anchored_target_label']} ({c_row['anchored_target_id']})")
    else:
        print("[WARNING] Could not parse any grammatical assertions from the available text files.")


if __name__ == "__main__":
    compile_hypergraph_claims()
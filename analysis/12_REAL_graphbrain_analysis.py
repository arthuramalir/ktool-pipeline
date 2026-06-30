from pathlib import Path
import pandas as pd
from graphbrain import hgraph, hedge
from graphbrain.parsers import create_parser

def run_native_graphbrain() -> None:
    print("🔮 Initializing Native ERC-Socsemics Graphbrain Engine...")
    
    # 1. Initialize a true, persistent SQLite-backed hypergraph database
    db_path = "data/urdabai_semantic_hypergraph.db"
    hg = hgraph(db_path)
    print(f"   Database initialized at: {db_path}")
    
    # 2. Spin up the deep linguistic Hypergraph Parser
    print("   Loading natural language parsing models (this may take 10-15 seconds)...")
    parser = create_parser(lang="en")
    
    # Let's pull your real raw sentences from the data
    sample_narratives = [
        "Liquid Therapy helped me overcome my fear of the ocean.",
        "Rethink Ireland helped Helium Arts reach a wider audience.",
        "They’re not the problem, they’re the evidence."
    ]
    
    print(f"\nParsing {len(sample_narratives)} sentences into true recursive hyperedges...")
    
    # 3. The Core Ingestion Loop
    for sentence in sample_narratives:
        parse_results = parser.parse(sentence)
        for par in parse_results["parses"]:
            main_edge = par["main_edge"]
            
            # This is a real Graphbrain Hyperedge object!
            print(f"\n📝 Original Text: \"{sentence}\"")
            print(f"   🤖 True Hyperedge: {main_edge}")
            
            # Commit it directly to the true hypergraph database
            hg.add(main_edge)
            
    print(f"\n💾 Database write complete. Total edges stored: {hg.edge_count()}")
    print("=" * 60)
    print("   RUNNING ADVANCED DEEP PATTERN MATCHING")
    print("=" * 60)
    
    # 4. Native Variable Binding (The real magic)
    # The '...' tells Graphbrain to allow recursive sub-clauses of any size 
    # to bind completely to our CATALYST and BENEFICIARY variables!
    pattern = "(helped/P CATALYST/C BENEFICIARY/C ...)"
    
    # Search the live hypergraph database using native non-strict pattern rules
    print(f"Querying pattern: {pattern}\n")
    matches = list(hg.match(pattern, strict=False))
    
    for edge, variables in matches:
        print(f"🔥 Rule Triggered on Edge: {edge}")
        # Extract the deep variables resolved by the engine
        catalyst = variables.get("CATALYST", "Unknown")
        beneficiary = variables.get("BENEFICIARY", "Unknown")
        
        print(f"     🔹 Bound [CATALYST]: {catalyst} (Type: {type(catalyst)})")
        print(f"     🔹 Bound [BENEFICIARY]: {beneficiary}")
        print("-" * 50)

if __name__ == "__main__":
    run_native_graphbrain()
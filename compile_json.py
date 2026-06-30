import json
import os
import pandas as pd

def build_deterministic_sensemaking_matrix():
    info_path = "data/173/informations.json"
    perc_path = "data/173/perceptions.json"
    output_dir = "data/processed/173"
    os.makedirs(output_dir, exist_ok=True)

    print("========================================================================")
    print("🧠 RUNNING ETHNOGRAPHIC MATRIX MATRIX COMPILER")
    print("========================================================================")

    if not os.path.exists(info_path) or not os.path.exists(perc_path):
        print("❌ Core raw datasets missing. Please verify your data directory.")
        return

    with open(info_path, "r", encoding="utf-8") as f: infos = json.load(f).get("data", [])
    with open(perc_path, "r", encoding="utf-8") as f: percs = json.load(f).get("data", [])

    print(f"📦 Assets Loaded: {len(infos)} Deep Qualitative Quotes | {len(percs)} Macro Pillars")

    # 1. Establish the Definitive K-Tool Semantic Taxonomy for Urdaibai
    # Calibrating vocabulary arrays to cross-reference raw ethnographic text entries
    taxonomy = {
        "Green Deal": [
            "environment", "green", "deal", "nature", "forest", "sustainable", "policy", 
            "respiratory", "pollution", "climate", "biodiversity", "conservation", "protected"
        ],
        "Infrastructure": [
            "isolated", "infrastructure", "transport", "roads", "north", "world", "rail",
            "train", "highway", "access", "connection", "mobility", "travel", "distance"
        ],
        "Economic development": [
            "business", "economic", "development", "funding", "market", "money", "industry",
            "tourism", "jobs", "employment", "investment", "agriculture", "local economy"
        ],
        "TCc in Europe": [
            "europe", "european", "tcc", "international", "brussels", "eu funding", "strasbourg"
        ],
        "TCc Implement": [
            "implement", "committee", "technical", "action", "government", "difficult", 
            "regulation", "administration", "execution", "bureaucracy", "local council"
        ],
        "Peacebuilding": [
            "peace", "conflict", "community", "building", "stability", "coexistence", 
            "dialogue", "reconciliation", "cohesion", "social fabric", "integration"
        ]
    }

    # 2. Process the Cohort
    final_records = []
    
    for item in infos:
        i_id = item.get("id")
        attrs = item.get("attributes", {})
        quote_text = attrs.get("quote", "").strip()
        
        # Calculate scores across the classification taxonomy
        best_match = "General Discourse / Cross-Cutting"
        highest_score = 0
        
        for category, keywords in taxonomy.items():
            score = sum(1 for word in keywords if word in quote_text.lower())
            if score > highest_score:
                highest_score = score
                best_match = category
                
        final_records.append({
            "quote_id": i_id,
            "reference_number": attrs.get("reference_number"),
            "date": attrs.get("date"),
            "locale": attrs.get("locale"),
            "assigned_pillar": best_match,
            "taxonomy_match_weight": highest_score,
            "quote_text": quote_text
        })

    # 3. Save Matrix and Output Results
    df_master = pd.DataFrame(final_records)
    df_master.to_csv(f"{output_dir}/master_ethnographic_matrix.csv", index=False)

    print("\n📊 MASTER DATA COMPILATION STATISTICS")
    print("========================================================================")
    print(f" • Total Clean Quotes Sorted: {len(df_master)} / {len(infos)}")
    print("\n📈 DISTRIBUTION BY STRATEGIC NARRATIVE PILLAR:")
    print("-" * 72)
    
    distribution = df_master["assigned_pillar"].value_counts()
    for category, count in distribution.items():
        percentage = (count / len(df_master)) * 100
        print(f"  ↳ {category:<35} : {count:>2} Quotes ({percentage:>5.1f}%)")

    print(f"\n💾 Clean analytical matrix saved to: {output_dir}/master_ethnographic_matrix.csv")

if __name__ == "__main__":
    build_deterministic_sensemaking_matrix()
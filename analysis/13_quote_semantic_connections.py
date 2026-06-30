import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PATH CONFIGURATION ---
PLATFORM_ID = "173"
OUTPUT_SUBDIR = "test"
BASE_DIR = f"data/processed/{PLATFORM_ID}/{OUTPUT_SUBDIR}"
INFORMATIONS_CSV = os.path.join(BASE_DIR, "entities", "informations.csv")
MASTER_EDGES_CSV = os.path.join(BASE_DIR, "analytics", "edges.csv")
OUTPUT_SEMANTIC_CSV = os.path.join(BASE_DIR, "relationships", "quote_semantic_edges.csv")

# --- ALC FRAMEWORK REGEX LEXICONS (Tightened to avoid filler-word false positives) ---
CAUSAL_CUES = r"\b(because|therefore|consequently|so that|led to|resulted in|due to|that's why|creates|generates)\b"
CONTRADICTION_WORDS = r"\b(oppose|against|disagree|criticise|criticize|protest|reject|deny|stop|harm|negative|wrong|fail|conflict)\b"

EMOTION_LEXICON = {
    "vehement": r"\b(angry|unacceptable|protest|fight|criticise|destroy|worst|demanding|stop|fair)\b",
    "nostalgic": r"\b(used to|past|before|history|remember|traditional|old days|lost|origin)\b",
    "resigned": r"\b(depend on|accept|nothing we can do|whatever|survive|cope|is what it is|limit)\b"
}

def parse_tags(tag_string):
    """Safely converts comma/pipe separated string tags into clean python sets"""
    if pd.isna(tag_string) or not isinstance(tag_string, str):
        return set()
    return {t.strip().lower() for t in re.split(r'[|,]', tag_string) if t.strip()}

def detect_intensity_emotion(text):
    """ALC Pattern Criterion: Determines qualitative narrative tone based on vocabulary intensity"""
    text_lower = str(text).lower()
    for emotion, pattern in EMOTION_LEXICON.items():
        if re.search(pattern, text_lower):
            return emotion
    return "balanced"

def generate_alc_comprehensive_edges(info_df):
    """
    Executes the 5 Parameters of Analysis and tags them with the 3 Core Analytical Criteria
    to structure raw quotes into an interconnected ALC Qualitative Grid.
    """
    print(f"-> Analyzing {len(info_df)} listening layer quotes against ALC parameters...")
    all_edges = []
    
    # Pre-clean data text strings
    info_df["clean_text"] = info_df["quote"].fillna(info_df["description"]).fillna("").astype(str)
    
    # Compute TF-IDF matrix for cross-channel similarity validation
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(info_df["clean_text"]) if len(info_df) > 0 else None
    
    # Calculate Frequency metrics at the platform level (Emerging Patterns)
    topic_counts = {}
    for _, row in info_df.iterrows():
        sub_areas = parse_tags(row.get("topics_sub_areas"))
        for sa in sub_areas:
            topic_counts.setdefault(sa, set()).add(row.get("channel_id"))
    
    # Mark a topic as an 'Emerging Pattern' if it appears across 4 or more independent channels
    emerging_topics = {topic for topic, channels in topic_counts.items() if len(channels) >= 4}

    # =========================================================================
    # PARAMETER 1: SEQUENCE (Within same interview channel)
    # =========================================================================
    grouped_channels = info_df.groupby("channel_id")
    for channel_id, group in grouped_channels:
        if not channel_id or pd.isna(channel_id):
            continue
        sorted_group = group.sort_values(by=["date", "native_id"])
        
        for i in range(len(sorted_group) - 1):
            src = sorted_group.iloc[i]
            tgt = sorted_group.iloc[i + 1]
            
            all_edges.append({
                "source": src["global_id"],
                "target": tgt["global_id"],
                "connection_type": "sequence",
                "parameter": "Sequential Timeline",
                "weight": 2.0,
                "intensity_emotion": detect_intensity_emotion(src["clean_text"]),
                "narrative_coalition": "No (Intra-Interview Flow)",
                "evidence_source": "Chronological interview chain",
                "description": f"Narrative sequence trace in channel {src.get('channel_code')}"
            })

    # =========================================================================
    # PARAMETERS 2, 3, 4, 5: CROSS-CHANNEL SYNTHESIS (Inter-Interview connections)
    # =========================================================================
    info_list = info_df.to_dict('records')
    
    for i in range(len(info_list)):
        for j in range(i + 1, len(info_list)):
            node_a = info_list[i]
            node_b = info_list[j]
            
            if node_a["channel_id"] == node_b["channel_id"]:
                continue
                
            tags_a = parse_tags(node_a.get("topics_sub_areas"))
            tags_b = parse_tags(node_b.get("topics_sub_areas"))
            shared_topics = tags_a.intersection(tags_b)
            
            if not shared_topics:
                continue
                
            text_a = node_a["clean_text"]
            text_b = node_b["clean_text"]
            
            # Calculate vocabulary cosine overlap index
            cosine_sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0] if tfidf_matrix is not None else 0.0
            
            emotion_a = detect_intensity_emotion(text_a)
            emotion_b = detect_intensity_emotion(text_b)
            resolved_intensity = emotion_a if emotion_a == emotion_b else f"{emotion_a}-{emotion_b}"
            
            # Target features
            has_causal_cue = bool(re.search(CAUSAL_CUES, text_a.lower()) or re.search(CAUSAL_CUES, text_b.lower()))
            has_explicit_conflict = bool(re.search(CONTRADICTION_WORDS, text_a.lower()) or re.search(CONTRADICTION_WORDS, text_b.lower()))

            # --- EXPERT ALC HIERARCHY CLASSIFICATION ENGINE ---
            
            # Rule A: Causal Chain Tracing
            if has_causal_cue and cosine_sim > 0.08:
                all_edges.append({
                    "source": node_a["global_id"],
                    "target": node_b["global_id"],
                    "connection_type": "causality",
                    "parameter": "Causal Link",
                    "weight": 2.5,
                    "intensity_emotion": resolved_intensity,
                    "narrative_coalition": "Potential",
                    "evidence_source": "NLP Causal Cue Overlap",
                    "description": f"Cause-and-effect narrative link regarding: {', '.join(shared_topics)}"
                })
                
            # Rule B: High/Moderate Semantic Overlap -> Similarity/Frequency Match
            elif cosine_sim > 0.12:
                is_emerging = any(topic in emerging_topics for topic in shared_topics)
                conn_type = "frequency" if is_emerging else "similarity"
                param_label = "Frequency (Emerging Pattern)" if is_emerging else "Similarity (Match)"
                is_coalition = "Yes" if cosine_sim > 0.22 else "Potential"
                
                all_edges.append({
                    "source": node_a["global_id"],
                    "target": node_b["global_id"],
                    "connection_type": conn_type,
                    "parameter": param_label,
                    "weight": 3.0 if is_emerging else 1.8,
                    "intensity_emotion": resolved_intensity,
                    "narrative_coalition": is_coalition,
                    "evidence_source": f"Cosine Overlap: {round(cosine_sim, 2)}",
                    "description": f"Thematic alignment on '{', '.join(shared_topics)}'"
                })
                
            # Rule C: Shared Topic + Low Vocabulary Overlap + Conflict Words -> True Contradiction
            elif has_explicit_conflict:
                all_edges.append({
                    "source": node_a["global_id"],
                    "target": node_b["global_id"],
                    "connection_type": "contradiction",
                    "parameter": "Divergent / Opposition Viewpoints",
                    "weight": 1.2,
                    "intensity_emotion": resolved_intensity,
                    "narrative_coalition": "No (Systemic Friction Point)",
                    "evidence_source": "Explicit Friction Overlap Check",
                    "description": f"Contradiction identified on topic '{list(shared_topics)[0]}'"
                })

    return pd.DataFrame(all_edges)

def merge_into_master_edges(quote_edges_df):
    """Safely appends the newly computed narrative connections into the central matrix file"""
    if quote_edges_df.empty:
        print("Warning: No new semantic edges were generated to merge.")
        return

    if os.path.exists(MASTER_EDGES_CSV):
        print(f"-> Reading existing master structural matrix from: {MASTER_EDGES_CSV}")
        master_df = pd.read_csv(MASTER_EDGES_CSV)
    else:
        print(f"Notice: Master matrix file not found. Creating a fresh one.")
        master_df = pd.DataFrame()

    # Combine dataframes
    combined = pd.concat([master_df, quote_edges_df], ignore_index=True)

    # Clean out and recalculate fresh sequential network IDs
    if "edge_id" in combined.columns:
        combined = combined.drop(columns=["edge_id"])
    combined.insert(0, "edge_id", [f"edge_{i + 1}" for i in range(len(combined))])

    # Save outputs
    os.makedirs(os.path.dirname(MASTER_EDGES_CSV), exist_ok=True)
    combined.to_csv(MASTER_EDGES_CSV, index=False)
    print(f"=== SUCCESS: Master matrix successfully written with {len(combined)} total rows. ===")

def main():
    print("=========================================================")
    print("STARTING: ALC Quote Semantic Matrix Ingestion Engine")
    print("=========================================================")
    
    if not os.path.exists(INFORMATIONS_CSV):
        print(f"ERROR: Missing source file: {INFORMATIONS_CSV}")
        return

    info_df = pd.read_csv(INFORMATIONS_CSV)
    quote_edges_df = generate_alc_comprehensive_edges(info_df)
    
    if not quote_edges_df.empty:
        print("\n--- PROCESSED METRIC SUMMARY ---")
        print(quote_edges_df["parameter"].value_counts().to_string())
        print("--------------------------------")
        
        os.makedirs(os.path.dirname(OUTPUT_SEMANTIC_CSV), exist_ok=True)
        quote_edges_df.to_csv(OUTPUT_SEMANTIC_CSV, index=False)
        print(f"-> Standalone relationship log saved to: {OUTPUT_SEMANTIC_CSV}")
        
        merge_into_master_edges(quote_edges_df)
    else:
        print("No qualitative relationships identified matching current framework filters.")

if __name__ == "__main__":
    main()
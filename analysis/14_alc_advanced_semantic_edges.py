import os
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# --- PATH CONFIGURATION ---
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
BASE_DIR = f"data/processed/{PLATFORM_ID}/{OUTPUT_SUBDIR}"
INFORMATIONS_CSV = os.path.join(BASE_DIR, "entities", "informations.csv")
MASTER_EDGES_CSV = os.path.join(BASE_DIR, "analytics", "edges.csv")
OUTPUT_SEMANTIC_CSV = os.path.join(BASE_DIR, "relationships", "quote_semantic_edges.csv")

# Silence Hugging Face Symlink warnings on Windows platforms
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- SYSTEM COLUMN BLUEPRINT (EXACT SCHEMA MATCH) ---
MASTER_SCHEMA_COLUMNS = [
    "edge_id", "source_global_id", "target_global_id", "edge_type", "edge_family",
    "methodological_phase", "directed", "evidence_source", "platform_id", "source_agent_id",
    "target_kind", "connection_type", "weight", "source_initiative_global_id", "information_id",
    "value_id", "perception_id", "challenge_id", "challenge_name", "source",
    "target", "parameter", "intensity_emotion", "narrative_coalition", "description",
    "is_ai_generated", "edge_origin", "generated_by", "inference_method"
]

# --- ALC LEXICONS FOR HYBRID ENRICHMENT ---
CAUSAL_CUES = r"\b(because|therefore|consequently|so that|led to|resulted in|due to|that's why|creates|generates|subsidies only go to|small businesses are all closing)\b"

EMOTION_LEXICON = {
    "vehement": r"\b(angry|unacceptable|protest|fight|criticise|destroy|worst|demanding|stop|fair|longing|frustration)\b",
    "nostalgic": r"\b(used to|past|before|history|remember|traditional|old days|lost|origin|fishing|canning)\b",
    "resigned": r"\b(depend on|accept|nothing we can do|whatever|survive|cope|is what it is|limit|now we depend)\b"
}

def parse_tags(tag_string):
    """Safely converts comma/pipe separated string tags into clean python sets."""
    if pd.isna(tag_string) or not isinstance(tag_string, str):
        return set()
    return {t.strip().lower() for t in re.split(r'[|,]', tag_string) if t.strip()}

def detect_intensity_emotion(text):
    """ALC Pattern Criterion: Determines qualitative narrative tone based on vocabulary intensity."""
    text_lower = str(text).lower()
    for emotion, pattern in EMOTION_LEXICON.items():
        if re.search(pattern, text_lower):
            return emotion
    return "balanced"

def create_blank_edge():
    """Returns a fresh dictionary matching the exact schema signature required by KTool."""
    edge = {col: None for col in MASTER_SCHEMA_COLUMNS}
    edge.update({
        "is_ai_generated": True,
        "edge_origin": "ai_inferred",
        "generated_by": "14_alc_advanced_semantic_edges.py",
        "inference_method": "sentence_transformer_cross_encoder",
    })
    return edge

def generate_transformer_alc_edges(info_df, similarity_threshold=0.62):
    """
    Executes the 5 ALC Parameters of Analysis using pure English Bi-Encoders 
    and NLI Cross-Encoders, outputting a structurally clean master graph schema.
    """
    print(f"-> Initializing Deep Semantic Extraction on {len(info_df)} English elements...")
    all_edges = []
    
    info_df["clean_text"] = info_df["quote"].fillna(info_df["description"]).fillna("").astype(str)
    info_df = info_df[info_df["clean_text"].str.strip() != ""].reset_index(drop=True)
    
    if len(info_df) == 0:
        return pd.DataFrame(columns=MASTER_SCHEMA_COLUMNS)

    # 1. Load English Native Models
    print("-> Loading English Bi-Encoder for topical mapping...")
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("-> Loading NLI Cross-Encoder for deep contradiction detection...")
    nli_model = CrossEncoder("cross-encoder/nli-distilroberta-base")
    
    # 2. Vectorize Corpus
    corpus_text = info_df["clean_text"].tolist()
    embeddings = bi_encoder.encode(corpus_text, show_progress_bar=True)
    similarity_matrix = cosine_similarity(embeddings)
    
    # 3. Dynamic Thresholding Setup for Emerging Patterns
    topic_channels = {}
    for idx, row in info_df.iterrows():
        sub_areas = parse_tags(row.get("topics_sub_areas"))
        for sa in sub_areas:
            topic_channels.setdefault(sa, set()).add(row.get("channel_id"))
            
    min_channels_for_emerging = 3 if len(info_df) < 200 else 10
    emerging_topics = {topic for topic, channels in topic_channels.items() if len(channels) >= min_channels_for_emerging}
    
    # =========================================================================
    # PARAMETER 4: SEQUENCE (Sequential Timeline Within Same Interview Track)
    # =========================================================================
    print("-> Processing Parameter 4: Sequence (Sequential)...")
    grouped_channels = info_df.groupby("channel_id")
    for channel_id, group in grouped_channels:
        if not channel_id or pd.isna(channel_id):
            continue
        sorted_group = group.sort_values(by=["date", "native_id" if "native_id" in group.columns else "global_id"])
        
        for i in range(len(sorted_group) - 1):
            src = sorted_group.iloc[i]
            tgt = sorted_group.iloc[i + 1]
            
            edge = create_blank_edge()
            edge.update({
                "source_global_id": src["global_id"],
                "target_global_id": tgt["global_id"],
                "source": src["global_id"],
                "target": tgt["global_id"],
                "edge_type": "sequence",
                "connection_type": "sequence",
                "edge_family": "qualitative_narrative",
                "methodological_phase": "listening",
                "directed": True,
                "weight": 2.0,
                "evidence_source": "Chronological Tracking",
                "platform_id": PLATFORM_ID,
                "parameter": "Sequence (Sequential)",
                "intensity_emotion": detect_intensity_emotion(src["clean_text"]),
                "narrative_coalition": "No (Intra-Interview Flow)",
                "description": f"Narrative chain connection within interview channel {channel_id}."
            })
            all_edges.append(edge)

    # =========================================================================
    # PARAMETERS 1, 2, 3, 5: CROSS-CHANNEL SYNTHESIS
    # =========================================================================
    print("-> Processing Cross-Channel Parameters...")
    info_list = info_df.to_dict('records')
    
    for i in range(len(info_list)):
        for j in range(i + 1, len(info_list)):
            node_a = info_list[i]
            node_b = info_list[j]
            
            if node_a["channel_id"] == node_b["channel_id"]:
                continue
                
            semantic_score = float(similarity_matrix[i][j])
            tags_a = parse_tags(node_a.get("topics_sub_areas"))
            tags_b = parse_tags(node_b.get("topics_sub_areas"))
            shared_topics = tags_a.intersection(tags_b)
            
            if semantic_score >= similarity_threshold or len(shared_topics) > 0:
                text_a = node_a["clean_text"]
                text_b = node_b["clean_text"]
                
                emotion_a = detect_intensity_emotion(text_a)
                emotion_b = detect_intensity_emotion(text_b)
                resolved_emotion = emotion_a if emotion_a == emotion_b else f"{emotion_a}-{emotion_b}"
                
                has_causal_cue = bool(re.search(CAUSAL_CUES, text_a.lower()) or re.search(CAUSAL_CUES, text_b.lower()))
                
                # Evaluate Stance Logic using NLI Cross-Encoder
                nli_scores = nli_model.predict([(text_a, text_b)])[0]
                contradiction_prob = float(nli_scores[0])
                
                # Setup core framework fields
                edge = create_blank_edge()
                edge.update({
                    "source_global_id": node_a["global_id"],
                    "target_global_id": node_b["global_id"],
                    "source": node_a["global_id"],
                    "target": node_b["global_id"],
                    "edge_family": "qualitative_narrative",
                    "methodological_phase": "listening",
                    "platform_id": PLATFORM_ID,
                    "intensity_emotion": resolved_emotion
                })
                
                # --- PARAMETER 5: CAUSALITY (Causal) ---
                if semantic_score >= (similarity_threshold - 0.05) and has_causal_cue:
                    edge.update({
                        "edge_type": "causality",
                        "connection_type": "causality",
                        "directed": True,
                        "weight": round(semantic_score + 0.5, 4),
                        "evidence_source": f"Bi-Encoder: {round(semantic_score, 2)} + Causal Cue",
                        "parameter": "Causality (Causal)",
                        "narrative_coalition": "Potential Coalition Block",
                        "description": "One citation identifies the cause/consequence framework of another."
                    })
                    all_edges.append(edge)
                    
                # --- PARAMETER 2: DIFFERENCE (Contradictory) ---
                elif contradiction_prob > 0.55:
                    edge.update({
                        "edge_type": "contradiction",
                        "connection_type": "contradiction",
                        "directed": False,
                        "weight": round(contradiction_prob * 2, 4),
                        "evidence_source": f"NLI Contradiction Stance: {round(contradiction_prob, 2)}",
                        "parameter": "Difference (Contradictory)",
                        "intensity_emotion": "systemic-friction",
                        "narrative_coalition": "No (Systemic Friction Point)",
                        "description": f"Opposing viewpoints identified contextually on topics: {list(shared_topics) if shared_topics else 'Context alignment'}."
                    })
                    all_edges.append(edge)
                    
                # --- PARAMETERS 1 & 3: SIMILARITY (Match) & FREQUENCY (Emerging) ---
                elif semantic_score >= similarity_threshold:
                    is_emerging = any(topic in emerging_topics for topic in shared_topics) if shared_topics else False
                    
                    if is_emerging:
                        edge.update({
                            "edge_type": "frequency",
                            "connection_type": "frequency",
                            "directed": False,
                            "weight": round(semantic_score + 0.4, 4),
                            "evidence_source": f"Cosine: {round(semantic_score, 2)} (Emerging across channels)",
                            "parameter": "Frequency (Emerging)",
                            "narrative_coalition": "Yes (Broad Narrative Coalition)",
                            "description": f"Several independent citations repeating the same framework perception on: {list(shared_topics)}."
                        })
                    else:
                        edge.update({
                            "edge_type": "similarity",
                            "connection_type": "similarity",
                            "directed": False,
                            "weight": round(semantic_score, 4),
                            "evidence_source": f"Cosine: {round(semantic_score, 2)}",
                            "parameter": "Similarity (Match)",
                            "narrative_coalition": "Yes" if semantic_score > 0.72 else "Potential",
                            "description": "Connections between citations that express very similar ideas."
                        })
                    all_edges.append(edge)
                        
    return pd.DataFrame(all_edges)

def merge_into_master_edges(quote_edges_df):
    """Safely appends computed narrative connections into the central matrix file without duplication."""
    if quote_edges_df.empty:
        print("Warning: No new semantic edges were generated.")
        return

    if os.path.exists(MASTER_EDGES_CSV):
        master_df = pd.read_csv(MASTER_EDGES_CSV)
        valid_types = {"sequence", "causality", "contradiction", "frequency", "similarity"}
        if "is_ai_generated" in master_df.columns:
            ai_mask = master_df["is_ai_generated"].astype(str).str.lower().isin({"true", "1", "yes"})
        else:
            ai_mask = pd.Series(False, index=master_df.index)
        
        # Clean previous iterations to prevent file inflation
        if "edge_type" in master_df.columns:
            master_df = master_df[~(master_df["edge_type"].isin(valid_types) & ai_mask)]
        elif "connection_type" in master_df.columns:
            master_df = master_df[~(master_df["connection_type"].isin(valid_types) & ai_mask)]
    else:
        master_df = pd.DataFrame(columns=MASTER_SCHEMA_COLUMNS)

    # Combine historical edges with your new formatted edge data
    combined = pd.concat([master_df, quote_edges_df], ignore_index=True)

    # Generate sequential unique keys inside the proper system header
    if "edge_id" in combined.columns:
        combined = combined.drop(columns=["edge_id"])
    combined.insert(0, "edge_id", [f"narrative_edge_{i + 1}" for i in range(len(combined))])

    # Re-enforce explicit column alignment rules right before writing the file
    for col in MASTER_SCHEMA_COLUMNS:
        if col not in combined.columns:
            combined[col] = None
    passthrough_columns = [col for col in combined.columns if col not in MASTER_SCHEMA_COLUMNS]
    combined = combined[MASTER_SCHEMA_COLUMNS + passthrough_columns]

    os.makedirs(os.path.dirname(MASTER_EDGES_CSV), exist_ok=True)
    combined.to_csv(MASTER_EDGES_CSV, index=False)
    print(f"=== SUCCESS: Master graph matrix written with {len(combined)} total rows. ===")

def main():
    if not os.path.exists(INFORMATIONS_CSV):
        print(f"ERROR: Missing source file: {INFORMATIONS_CSV}")
        return

    info_df = pd.read_csv(INFORMATIONS_CSV)
    quote_edges_df = generate_transformer_alc_edges(info_df, similarity_threshold=0.62)
    
    if not quote_edges_df.empty:
        print("\n--- ETHNOGRAPHIC NARRATIVE METRIC SUMMARY ---")
        print(quote_edges_df["edge_type"].value_counts().to_string())
        print("--------------------------------------------")
        
        os.makedirs(os.path.dirname(OUTPUT_SEMANTIC_CSV), exist_ok=True)
        quote_edges_df.to_csv(OUTPUT_SEMANTIC_CSV, index=False)
        merge_into_master_edges(quote_edges_df)
    else:
        print("No matches met current embedding rules.")

if __name__ == "__main__":
    main()

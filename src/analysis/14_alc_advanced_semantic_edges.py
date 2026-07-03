import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
BASE_DIR = f"data/processed/{PLATFORM_ID}/{OUTPUT_SUBDIR}"
INFORMATIONS_CSV = os.path.join(BASE_DIR, "entities", "informations.csv")
MASTER_EDGES_CSV = os.path.join(BASE_DIR, "analytics", "edges.csv")
OUTPUT_SEMANTIC_CSV = os.path.join(BASE_DIR, "relationships", "quote_semantic_edges.csv")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.55
CONTRADICTION_THRESHOLD = 0.45
CAUSALITY_SIM_THRESHOLD = 0.35

MASTER_SCHEMA_COLUMNS = [
    "edge_id", "source_global_id", "target_global_id", "edge_type", "edge_family",
    "methodological_phase", "directed", "evidence_source", "platform_id", "source_agent_id",
    "target_kind", "connection_type", "weight", "source_initiative_global_id", "information_id",
    "value_id", "perception_id", "challenge_id", "challenge_name", "source",
    "target", "parameter", "intensity_emotion", "narrative_coalition", "description",
    "is_ai_generated", "edge_origin", "generated_by", "inference_method"
]

# --- Emotion / tone lexicons ---
EMOTION_LEXICON = {
    "vehement": r"\b(angry|unacceptable|protest|fight|criticise|destroy|worst|demanding|stop|fair|longing|frustration|urgent|crisis|desperate|unacceptable|outrage)\b",
    "nostalgic": r"\b(used to|past|before|history|remember|traditional|old days|lost|origin|fishing|canning|longing)\b",
    "resigned": r"\b(depend on|accept|nothing we can do|whatever|survive|cope|is what it is|limit|now we depend|overwhelmed)\b",
}

CAUSAL_MARKERS = re.compile(
    r'\b(because|therefore|due to|as a result|consequently|hence|thus|'
    r'causes?|led to|leads to|resulted in|results in|creates|created|'
    r'explains|the reason|why|so that|in order to|if.+then|'
    r'knock.?on effect|chain of events|contributes? to|triggers?)\b',
    re.IGNORECASE
)

# Positive / negative framing words for contradiction detection
POSITIVE_WORDS = {
    "success", "great", "positive", "benefit", "opportunity", "improve", "progress",
    "transformative", "flourished", "thrive", "confident", "hope", "proud",
    "breakthrough", "growth", "enhance", "strengthen", "brilliant", "empower",
    "supportive", "good", "excellent", "valuable", "fantastic", "wonderful",
}

NEGATIVE_WORDS = {
    "challenge", "struggle", "lack", "limited", "poverty", "anxiety", "depression",
    "trauma", "crisis", "fear", "stigma", "shame", "overwhelmed", "suffering",
    "refusal", "violence", "addiction", "suicide", "harm", "difficult", "barrier",
    "disadvantaged", "neglect", "crime", "murder", "desperate", "unacceptable",
    "worst", "frustration", "isolated", "lonely", "overlooked", "gap",
}


# =============================================================================
# HELPERS
# =============================================================================

def parse_tags(tag_string):
    if pd.isna(tag_string) or not isinstance(tag_string, str):
        return set()
    return {t.strip().lower() for t in re.split(r'[|,]', tag_string) if t.strip()}


def detect_emotion(text):
    text_lower = str(text).lower()
    for emotion, pattern in EMOTION_LEXICON.items():
        if re.search(pattern, text_lower):
            return emotion
    return "balanced"


def sentiment_score(text):
    text_lower = str(text).lower()
    words = set(re.findall(r'\w+', text_lower))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def has_causal_markers(text):
    return bool(CAUSAL_MARKERS.search(str(text)))


def create_blank_edge():
    edge = {col: None for col in MASTER_SCHEMA_COLUMNS}
    edge.update({
        "is_ai_generated": False,
        "edge_origin": "ai_inferred",
        "generated_by": "14_alc_advanced_semantic_edges.py",
        "inference_method": "similarity_heuristics",
    })
    return edge


# =============================================================================
# LOAD & EMBED
# =============================================================================

def load_and_prepare_data():
    info_df = pd.read_csv(INFORMATIONS_CSV)
    info_df["clean_text"] = (
        info_df["quote"].fillna(info_df["description"]).fillna("").astype(str)
    )
    info_df = info_df[info_df["clean_text"].str.strip() != ""].reset_index(drop=True)
    return info_df


def compute_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    sim_matrix = cosine_similarity(embeddings)
    return embeddings, sim_matrix


# =============================================================================
# CANDIDATE GENERATION
# =============================================================================

def generate_candidates(info_list, sim_matrix):
    n = len(info_list)
    cross_candidates = set()
    within_sequence = []

    for i in range(n):
        ch_i = info_list[i].get("channel_id")
        tags_i = parse_tags(info_list[i].get("topics_sub_areas"))
        scored = []
        for j in range(n):
            if i == j:
                continue
            ch_j = info_list[j].get("channel_id")
            sem_score = float(sim_matrix[i][j])
            tags_j = parse_tags(info_list[j].get("topics_sub_areas"))
            shared_tags = bool(tags_i.intersection(tags_j))
            same_channel = (ch_i == ch_j and not pd.isna(ch_i))

            if same_channel:
                continue

            scored.append((j, sem_score, shared_tags))

        for j, sem_score, shared_tags in scored:
            if shared_tags or sem_score >= SIMILARITY_THRESHOLD:
                cross_candidates.add((min(i, j), max(i, j), "general"))

        scored.sort(key=lambda x: -x[1])
        for j, _, _ in scored[:TOP_K]:
            cross_candidates.add((min(i, j), max(i, j), "general"))

    # Within-channel sequence candidates (chronological)
    channels = {}
    for idx, rec in enumerate(info_list):
        ch = rec.get("channel_id")
        if pd.isna(ch) or ch is None:
            ch = "none"
        channels.setdefault(str(ch), []).append((idx, str(rec.get("date", ""))))

    for ch, members in channels.items():
        members.sort(key=lambda x: x[1])
        for k in range(len(members) - 1):
            i = members[k][0]
            j = members[k + 1][0]
            within_sequence.append((i, j))

    return list(cross_candidates), within_sequence


# =============================================================================
# EDGE BUILDING
# =============================================================================

def make_edge(node_a, node_b, edge_type, weight, description, directed=False):
    emotion_a = detect_emotion(node_a["clean_text"])
    emotion_b = detect_emotion(node_b["clean_text"])
    resolved_emotion = emotion_a if emotion_a == emotion_b else f"{emotion_a}-{emotion_b}"

    edge = create_blank_edge()
    edge.update({
        "source_global_id": node_a["global_id"],
        "target_global_id": node_b["global_id"],
        "edge_type": edge_type,
        "edge_family": "qualitative_narrative",
        "methodological_phase": "listening",
        "directed": directed,
        "weight": round(weight, 4),
        "evidence_source": f"cosine_sim rule-based inference",
        "platform_id": PLATFORM_ID,
        "intensity_emotion": resolved_emotion,
        "parameter": f"{edge_type.title()} (Rule)",
        "narrative_coalition": "Yes" if weight > 0.8 else "Potential",
        "description": description,
    })
    return edge


# =============================================================================
# CLASSIFIER — rule-based, no LLM
# =============================================================================

def classify_cross_pair(node_a, node_b, sim_score):
    text_a = (node_a["clean_text"] or "").lower()
    text_b = (node_b["clean_text"] or "").lower()
    tags_a = parse_tags(node_a.get("topics_sub_areas"))
    tags_b = parse_tags(node_b.get("topics_sub_areas"))
    shared_tags = bool(tags_a.intersection(tags_b))
    sent_a = sentiment_score(text_a)
    sent_b = sentiment_score(text_b)

    # --- Contradiction ---
    if shared_tags and sim_score >= CONTRADICTION_THRESHOLD:
        if sent_a * sent_b < -0.1:
            if sent_a > sent_b:
                desc = f"Quote A expresses positive view while Quote B expresses negative view on same topic ({', '.join(tags_a & tags_b) if tags_a & tags_b else 'shared topic'})."
            else:
                desc = f"Quote B expresses positive view while Quote A expresses negative view on same topic ({', '.join(tags_a & tags_b) if tags_a & tags_b else 'shared topic'})."
            weight = 0.5 + 0.5 * abs(sent_a - sent_b)
            return ("contradiction", round(min(weight, 0.95), 4), desc)

    # --- Causality ---
    if sim_score >= CAUSALITY_SIM_THRESHOLD:
        ca = has_causal_markers(text_a)
        cb = has_causal_markers(text_b)
        if ca and not cb:
            desc = f"Quote A uses causal language explaining why or how, and Quote B describes the resulting situation."
            return ("causality", round(0.5 + 0.5 * sim_score, 4), desc)
        if cb and not ca:
            desc = f"Quote B uses causal language explaining why or how, and Quote A describes the resulting situation."
            return ("causality", round(0.5 + 0.5 * sim_score, 4), desc)

    # --- Similarity ---
    if sim_score >= SIMILARITY_THRESHOLD or shared_tags:
        common_themes = tags_a & tags_b
        theme_str = f" on {', '.join(common_themes)}" if common_themes else ""
        desc = f"Both quotes express similar views about the same topic{theme_str}."
        weight = 0.5 + 0.5 * sim_score
        return ("similarity", round(min(weight, 0.95), 4), desc)

    return ("none", 0.0, "")


def classify_sequence_pair(node_a, node_b):
    text_a = (node_a["clean_text"] or "").lower()
    text_b = (node_b["clean_text"] or "").lower()
    tags_a = parse_tags(node_a.get("topics_sub_areas"))
    tags_b = parse_tags(node_b.get("topics_sub_areas"))
    shared_tags = bool(tags_a.intersection(tags_b))

    same_topic = shared_tags or sentiment_score(text_a) * sentiment_score(text_b) != 0
    if same_topic:
        desc = f"Quote A introduces a theme that Quote B continues or expands upon within the same channel."
        return ("sequence", 0.85, desc)

    desc = f"Quote A narratively leads into Quote B within the same channel (chronological order)."
    return ("sequence", 0.65, desc)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline():
    print("=" * 60)
    print(f"ALC Semantic Edge Pipeline — Platform {PLATFORM_ID}")
    print("Rule-based (no LLM)")
    print("=" * 60)

    info_df = load_and_prepare_data()
    print(f"Loaded {len(info_df)} information nodes.")

    embeddings, sim_matrix = compute_embeddings(info_df["clean_text"].tolist())
    info_list = info_df.to_dict("records")
    cross_candidates, sequence_candidates = generate_candidates(info_list, sim_matrix)

    print(f"Cross-channel candidates: {len(cross_candidates)}")
    print(f"Within-channel sequence candidates: {len(sequence_candidates)}")

    edges = []
    counts = {}

    for i, j, _ in cross_candidates:
        node_a, node_b = info_list[i], info_list[j]
        sim_score = float(sim_matrix[i][j])
        label, weight, desc = classify_cross_pair(node_a, node_b, sim_score)
        if label == "none":
            continue
        directed = label == "causality"
        edge = make_edge(node_a, node_b, label, weight, desc, directed=directed)
        edges.append(edge)
        counts[label] = counts.get(label, 0) + 1

    for i, j in sequence_candidates:
        node_a, node_b = info_list[i], info_list[j]
        label, weight, desc = classify_sequence_pair(node_a, node_b)
        edge = make_edge(node_a, node_b, label, weight, desc, directed=True)
        edges.append(edge)
        counts[label] = counts.get(label, 0) + 1

    print("\n--- Classification Summary ---")
    for label in ["similarity", "contradiction", "causality", "sequence"]:
        c = counts.get(label, 0)
        if c:
            print(f"  {label}: {c}")
    print(f"  total edges: {len(edges)}")
    print("-----------------------------")

    return pd.DataFrame(edges)


def merge_into_master_edges(quote_edges_df):
    if quote_edges_df.empty:
        print("Warning: No semantic edges were generated.")
        return

    if os.path.exists(MASTER_EDGES_CSV):
        master_df = pd.read_csv(MASTER_EDGES_CSV)
        valid_types = {"similarity", "contradiction", "causality", "sequence"}
        ai_mask = master_df["is_ai_generated"].astype(str).str.lower().isin({"true", "1", "yes"})
        if "edge_type" in master_df.columns:
            master_df = master_df[~(master_df["edge_type"].isin(valid_types) & ai_mask)]
    else:
        master_df = pd.DataFrame(columns=MASTER_SCHEMA_COLUMNS)

    combined = pd.concat([master_df, quote_edges_df], ignore_index=True)

    if "edge_id" in combined.columns:
        combined = combined.drop(columns=["edge_id"])
    combined.insert(0, "edge_id", [f"narrative_edge_{i + 1}" for i in range(len(combined))])

    for col in MASTER_SCHEMA_COLUMNS:
        if col not in combined.columns:
            combined[col] = None
    passthrough = [c for c in combined.columns if c not in MASTER_SCHEMA_COLUMNS]
    combined = combined[MASTER_SCHEMA_COLUMNS + passthrough]

    os.makedirs(os.path.dirname(MASTER_EDGES_CSV), exist_ok=True)
    combined.to_csv(MASTER_EDGES_CSV, index=False)
    print(f"=== Master graph written: {len(combined)} total edges. ===")


def main():
    if not os.path.exists(INFORMATIONS_CSV):
        print(f"ERROR: Missing {INFORMATIONS_CSV}")
        return

    quote_edges_df = run_pipeline()
    if not quote_edges_df.empty:
        print("\n--- Final Edge Type Counts ---")
        print(quote_edges_df["edge_type"].value_counts().to_string())
        print("------------------------------")
        os.makedirs(os.path.dirname(OUTPUT_SEMANTIC_CSV), exist_ok=True)
        quote_edges_df.to_csv(OUTPUT_SEMANTIC_CSV, index=False)
        merge_into_master_edges(quote_edges_df)
    else:
        print("No semantic edges generated.")


if __name__ == "__main__":
    main()

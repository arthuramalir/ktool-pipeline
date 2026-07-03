import os, re, sys
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "10")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
BASE_DIR = f"data/processed/{PLATFORM_ID}/{OUTPUT_SUBDIR}"
INFORMATIONS_CSV = os.path.join(BASE_DIR, "entities", "informations.csv")
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

# --- Spanish emotion / tone lexicons ---
EMOTION_LEXICON = {
    "vehement": r"\b(enojad[oa]|inaceptable|protesta|luch[ao]|criticar|destruir|terrible|exigiendo|frustraci[oó]n|urgente|crisis|desesperad[ao]|indignante|rabia|furia|insostenible|intolerable|alarmante)\b",
    "nostalgic": r"\b(sol[íi]a|pasado|antes|historia|recordar|tradicional|antiguo|viejos tiempos|perd[íi]do|origen|a[nñ]oranza|ra[íi]ces|costumbre|otro tiempo)\b",
    "resigned": r"\b(depende de|aceptar|no podemos hacer nada|como sea|sobrevivir|conforme|es lo que hay|l[íi]mite|ahora dependemos|abrumad[oa]|resignad[oa]|no hay opci[oó]n|to[s]ca aceptar)\b",
}

CAUSAL_MARKERS = re.compile(
    r'\b(porque|por lo tanto|debido a|como resultado|en consecuencia|por consiguiente|por eso|'
    r'causa|caus[óo]|provoca|provoc[óo]|genera|gener[óo]|lleva a|llev[óo] a|conduce a|'
    r'explica|la raz[oó]n|por qu[ée]|para que|a fin de|con el fin de|'
    r'contribuye a|contribuy[óo] a|desencadena|desencaden[óo]|produce|produjo|'
    r'motivo|raz[oó]n|origen|consecuencia|efecto|implica|implic[óo]|'
    r'da lugar a|trae como consecuencia|deriva en|resulta en)\b',
    re.IGNORECASE
)

POSITIVE_WORDS = {
    "exito", "éxito", "excelente", "positivo", "beneficio", "oportunidad", "mejorar",
    "progreso", "transformador", "florecer", "prosperar", "confianza", "esperanza",
    "orgulloso", "avance", "crecimiento", "fortalecer", "brillante", "empoderar",
    "apoyo", "bueno", "fantástico", "valioso", "innovador", "sostenible", "inclusivo",
    "participación", "colaboración", "efectivo", "eficiente", "solución",
}

NEGATIVE_WORDS = {
    "desafío", "lucha", "falta", "limitado", "pobreza", "ansiedad", "depresión",
    "trauma", "crisis", "miedo", "estigma", "vergüenza", "abrumado", "sufrimiento",
    "rechazo", "violencia", "adicción", "suicidio", "daño", "difícil", "barrera",
    "desfavorecido", "negligencia", "crimen", "asesinato", "desesperado",
    "inaceptable", "peor", "frustración", "aislado", "solitario", "ignorado",
    "brecha", "fracaso", "exclusión", "discriminación", "desigualdad", "injusticia",
    "corrupción", "abuso", "amenaza", "riesgo", "vulnerable", "marginado",
}


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
        "is_ai_generated": True,
        "edge_origin": "ai_inferred",
        "generated_by": "14_semantic_edges_p10.py",
        "inference_method": "multilingual_sentence_transformer",
    })
    return edge


def load_and_prepare_data():
    info_df = pd.read_csv(INFORMATIONS_CSV)
    info_df["clean_text"] = (
        info_df["quote"].fillna(info_df["description"]).fillna("").astype(str)
    )
    info_df = info_df[info_df["clean_text"].str.strip() != ""].reset_index(drop=True)
    return info_df


def compute_embeddings(texts):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    sim_matrix = cosine_similarity(embeddings)
    return embeddings, sim_matrix


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
        "evidence_source": "cosine_sim rule-based inference",
        "platform_id": PLATFORM_ID,
        "intensity_emotion": resolved_emotion,
        "parameter": f"{edge_type.title()} (Rule)",
        "narrative_coalition": "Yes" if weight > 0.8 else "Potential",
        "description": description,
    })
    return edge


def classify_cross_pair(node_a, node_b, sim_score):
    text_a = (node_a["clean_text"] or "").lower()
    text_b = (node_b["clean_text"] or "").lower()
    tags_a = parse_tags(node_a.get("topics_sub_areas"))
    tags_b = parse_tags(node_b.get("topics_sub_areas"))
    shared_tags = bool(tags_a.intersection(tags_b))
    sent_a = sentiment_score(text_a)
    sent_b = sentiment_score(text_b)

    if shared_tags and sim_score >= CONTRADICTION_THRESHOLD:
        if sent_a * sent_b < -0.1:
            if sent_a > sent_b:
                desc = f"Quote A expresses positive view while Quote B expresses negative view on same topic ({', '.join(tags_a & tags_b) if tags_a & tags_b else 'shared topic'})."
            else:
                desc = f"Quote B expresses positive view while Quote A expresses negative view on same topic ({', '.join(tags_a & tags_b) if tags_a & tags_b else 'shared topic'})."
            weight = 0.5 + 0.5 * abs(sent_a - sent_b)
            return ("contradiction", round(min(weight, 0.95), 4), desc)

    if sim_score >= CAUSALITY_SIM_THRESHOLD:
        ca = has_causal_markers(text_a)
        cb = has_causal_markers(text_b)
        if ca and not cb:
            desc = "Quote A uses causal language explaining why or how, and Quote B describes the resulting situation."
            return ("causality", round(0.5 + 0.5 * sim_score, 4), desc)
        if cb and not ca:
            desc = "Quote B uses causal language explaining why or how, and Quote A describes the resulting situation."
            return ("causality", round(0.5 + 0.5 * sim_score, 4), desc)

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
        desc = "Quote A introduces a theme that Quote B continues or expands upon within the same channel."
        return ("sequence", 0.85, desc)

    desc = "Quote A narratively leads into Quote B within the same channel (chronological order)."
    return ("sequence", 0.65, desc)


def run_pipeline():
    print("=" * 60)
    print(f"Semantic Edge Pipeline — Platform {PLATFORM_ID} (Chile COPOLAD)")
    print("Multilingual model + Spanish lexicons (no LLM)")
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

    return pd.DataFrame(edges)


def main():
    if not os.path.exists(INFORMATIONS_CSV):
        print(f"ERROR: Missing {INFORMATIONS_CSV}")
        return

    quote_edges_df = run_pipeline()
    if not quote_edges_df.empty:
        print(f"\n--- Final Edge Type Counts ---")
        print(quote_edges_df["edge_type"].value_counts().to_string())
        os.makedirs(os.path.dirname(OUTPUT_SEMANTIC_CSV), exist_ok=True)
        quote_edges_df.to_csv(OUTPUT_SEMANTIC_CSV, index=False)
        print(f"Saved: {OUTPUT_SEMANTIC_CSV} ({len(quote_edges_df)} edges)")
    else:
        print("No semantic edges generated.")


if __name__ == "__main__":
    main()

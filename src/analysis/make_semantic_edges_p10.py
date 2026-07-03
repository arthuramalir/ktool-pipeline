"""Create semantic edges for Platform 10 (Chile COPOLAD, Spanish).
Reads quote CSV, computes multilingual embeddings, classifies edges,
writes quote_semantic_edges.csv and merges into master edges.csv.

Usage: python src/analysis/make_semantic_edges_p10.py
"""

import os, re, sys, json
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "10" / "test"
INFO_CSV = DATA / "entities" / "informations.csv"
OUT_CSV = DATA / "relationships" / "quote_semantic_edges.csv"
EDGES_CSV = DATA / "analytics" / "edges.csv"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ── thresholds (tuned for multilingual model 768-dim) ──
SIM_T = 0.75
CONTRA_T = 0.65
CAUSE_T = 0.60
TOP_K = 3

# ── Spanish lexicons ──
CAUSAL_RE = re.compile(
    r'\b(porque|por lo tanto|debido a|como resultado|en consecuencia|por consiguiente|'
    r'causa|caus[óo]|provoca|provoc[óo]|genera|gener[óo]|lleva a|llev[óo] a|conduce a|'
    r'explica|la raz[oó]n|por qu[ée]|para que|a fin de|contribuye a|desencadena|'
    r'desencaden[óo]|produce|produjo|motivo|consecuencia|efecto|implica|implic[óo]|'
    r'da lugar a|resulta en|deriva en|trae como consecuencia)\b',
    re.IGNORECASE)

POS = {"exito","éxito","excelente","positivo","beneficio","oportunidad","mejorar",
       "progreso","transformador","florecer","prosperar","confianza","esperanza",
       "orgulloso","avance","crecimiento","fortalecer","brillante","empoderar",
       "apoyo","bueno","fantástico","valioso","innovador","sostenible","inclusivo",
       "participación","colaboración","efectivo","eficiente","solución"}

NEG = {"desafío","lucha","falta","limitado","pobreza","ansiedad","depresión",
       "trauma","crisis","miedo","estigma","vergüenza","abrumado","sufrimiento",
       "rechazo","violencia","adicción","suicidio","daño","difícil","barrera",
       "desfavorecido","negligencia","crimen","asesinato","desesperado",
       "inaceptable","peor","frustración","aislado","solitario","ignorado",
       "brecha","fracaso","exclusión","discriminación","desigualdad","injusticia",
       "corrupción","abuso","amenaza","riesgo","vulnerable","marginado"}

def parse_tags(s):
    if pd.isna(s) or not isinstance(s, str):
        return set()
    return {t.strip().lower() for t in re.split(r'[|,]', s) if t.strip()}

def sent_score(text):
    words = set(re.findall(r'\w+', str(text).lower()))
    pos = len(words & POS)
    neg = len(words & NEG)
    t = pos + neg
    return 0.0 if t == 0 else (pos - neg) / t

def has_causal(text):
    return bool(CAUSAL_RE.search(str(text)))

# ── Load data ──
print("Reading", INFO_CSV)
df = pd.read_csv(INFO_CSV)
df["text"] = df["quote"].fillna(df["description"]).fillna("").astype(str)
df = df[df["text"].str.strip() != ""].reset_index(drop=True)
print(f"  {len(df)} information nodes loaded")

# ── Embed (multilingual model) ──
print("Loading multilingual sentence transformer...")
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
print("Encoding...")
emb = model.encode(df["text"].tolist(), show_progress_bar=True)
sim = cosine_similarity(emb)
print(f"  Similarity matrix: {sim.shape}")

# ── Generate candidates ──
info = df.to_dict("records")
n = len(info)

cross = set()
seq = []
for i in range(n):
    ch_i = info[i].get("channel_id")
    tags_i = parse_tags(info[i].get("topics_sub_areas"))
    scored = []
    for j in range(n):
        if i == j:
            continue
        ch_j = info[j].get("channel_id")
        s = float(sim[i][j])
        shared = bool(tags_i & parse_tags(info[j].get("topics_sub_areas")))
        same_ch = (ch_i == ch_j and not pd.isna(ch_i))
        if same_ch:
            continue
        scored.append((j, s, shared))
    # Take TOP_K highest-similarity cross-channel candidates only
    scored.sort(key=lambda x: -x[1])
    for j, s, shared in scored[:TOP_K]:
        cross.add((min(i,j), max(i,j)))

# sequence candidates (within-channel chronological)
chan = {}
for idx, r in enumerate(info):
    c = r.get("channel_id")
    if pd.isna(c) or c is None:
        c = "none"
    chan.setdefault(str(c), []).append((idx, str(r.get("date",""))))
for ch, members in chan.items():
    members.sort(key=lambda x: x[1])
    for k in range(len(members)-1):
        seq.append((members[k][0], members[k+1][0]))

print(f"  Cross-channel candidates: {len(cross)}")
print(f"  Sequence candidates: {len(seq)}")

# ── Classify ──
def classify(a, b, s):
    ta, tb = a["text"].lower(), b["text"].lower()
    taga = parse_tags(a.get("topics_sub_areas"))
    tagb = parse_tags(b.get("topics_sub_areas"))
    shared = bool(taga & tagb)
    sa, sb = sent_score(ta), sent_score(tb)

    if shared and s >= CONTRA_T and sa * sb < -0.1:
        w = round(min(0.5 + 0.5*abs(sa-sb), 0.95), 4)
        if sa > sb:
            return "contradiction", w, f"A positive, B negative on shared topic"
        else:
            return "contradiction", w, f"B positive, A negative on shared topic"

    ca, cb = has_causal(ta), has_causal(tb)
    if s >= CAUSE_T:
        if ca and not cb:
            return "causality", round(0.5+0.5*s,4), "A explains causal driver, B describes result"
        if cb and not ca:
            return "causality", round(0.5+0.5*s,4), "B explains causal driver, A describes result"

    if s >= SIM_T or shared:
        w = round(min(0.5+0.5*s, 0.95), 4)
        return "similarity", w, "Similar views on same topic"

    return None, 0, ""

edges = []
counts = {}

for i,j in cross:
    s = float(sim[i][j])
    label, w, desc = classify(info[i], info[j], s)
    if label is None:
        continue
    e = {
        "source_global_id": info[i]["global_id"],
        "target_global_id": info[j]["global_id"],
        "edge_type": label,
        "edge_family": "qualitative_narrative",
        "methodological_phase": "listening",
        "directed": label == "causality",
        "weight": w,
        "evidence_source": "multilingual_cosine_rule",
        "platform_id": "10",
        "is_ai_generated": True,
        "edge_origin": "ai_inferred",
        "generated_by": "make_semantic_edges_p10.py",
        "inference_method": "multilingual_sentence_transformer",
        "description": desc,
    }
    edges.append(e)
    counts[label] = counts.get(label, 0) + 1

for i,j in seq:
    a_tags = parse_tags(info[i].get("topics_sub_areas"))
    b_tags = parse_tags(info[j].get("topics_sub_areas"))
    same_topic = bool(a_tags & b_tags) or sent_score(info[i]["text"]) * sent_score(info[j]["text"]) != 0
    w = 0.85 if same_topic else 0.65
    desc = "Same-channel continuation on same topic" if same_topic else "Chronological sequence within channel"
    e = {
        "source_global_id": info[i]["global_id"],
        "target_global_id": info[j]["global_id"],
        "edge_type": "sequence",
        "edge_family": "qualitative_narrative",
        "methodological_phase": "listening",
        "directed": True,
        "weight": w,
        "evidence_source": "multilingual_cosine_rule",
        "platform_id": "10",
        "is_ai_generated": True,
        "edge_origin": "ai_inferred",
        "generated_by": "make_semantic_edges_p10.py",
        "inference_method": "multilingual_sentence_transformer",
        "description": desc,
    }
    edges.append(e)
    counts["sequence"] = counts.get("sequence", 0) + 1

print("\n── Classification ──")
for label in ["similarity","contradiction","causality","sequence"]:
    print(f"  {label}: {counts.get(label,0)}")
print(f"  total: {len(edges)}")

# ── Save ──
out_df = pd.DataFrame(edges)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(out_df)} edges → {OUT_CSV}")

# ── Merge into master edges ──
if EDGES_CSV.exists():
    master = pd.read_csv(EDGES_CSV)
    sem_types = {"similarity","contradiction","causality","sequence"}
    ai_mask = master["is_ai_generated"].astype(str).str.lower().isin(["true","1","yes"])
    if "edge_type" in master.columns:
        master = master[~(master["edge_type"].isin(sem_types) & ai_mask)]
    combined = pd.concat([master, out_df], ignore_index=True)
    if "edge_id" in combined.columns:
        combined = combined.drop(columns=["edge_id"])
    combined.insert(0, "edge_id", [f"e{i+1}" for i in range(len(combined))])
    combined.to_csv(EDGES_CSV, index=False)
    print(f"Merged → {EDGES_CSV}: {len(combined)} total edges")
else:
    print(f"Master {EDGES_CSV} not found — edges saved to {OUT_CSV} only")

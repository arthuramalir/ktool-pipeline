"""Three-level narrative extraction using hyperbase semantic hypergraphs.

Extracts surface claims, implicit claims, and metanarratives from
information, perception, challenge, and value node texts.

Usage:
    set KTOOL_PLATFORM_ID=173 & set KTOOL_OUTPUT_SUBDIR=test
    python src/analysis/21_extract_narrative_layers.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
ANALYSIS_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR / "analysis"
ANALYTICS_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR / "analytics"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ── Implicit claim detection lexicons ──────────────────────────────────────

NEGATION_MARKERS = re.compile(
    r"\b(not|n't|never|no|none|nothing|nobody|nowhere|neither|nor|"
    r"lack of|absence of|without|fails? to|refuses? to|unable to)\b",
    re.IGNORECASE,
)

CONDITIONAL_MARKERS = re.compile(
    r"\b(if|unless|provided that|as long as|assuming|"
    r"in case|should it|were it|had it)\b",
    re.IGNORECASE,
)

EMERGENCY_MARKERS = re.compile(
    r"\b(crisis|urgent|desperate|critical|at risk|vulnerable|"
    r"broken|failing|collaps|unsustainable|cannot continue|"
    r"threaten|endanger|at breaking point)\b",
    re.IGNORECASE,
)

# ── Metanarrative / value dimension lexicons ────────────────────────────

VALUE_DIMENSIONS = {
    "social_justice": {
        "keywords": [
            "equity", "equality", "inclusion", "marginalis", "disadvantaged",
            "vulnerable", "deprivation", "poverty", "inequality", "fairness",
            "social justice", "human rights", "access", "discrimination",
            "minority", "ethnic", "traveller", "refugee", "asylum",
            "lgbtq", "disability", "special needs", "underserved",
        ],
        "belief_level": "deep_core",
    },
    "community_autonomy": {
        "keywords": [
            "community-led", "bottom-up", "local control", "grassroots",
            "community ownership", "self-determination", "empowerment",
            "community development", "local knowledge", "participatory",
            "co-design", "citizen-led", "neighbourhood", "parish",
            "volunteer", "local decision", "subsidiarity",
        ],
        "belief_level": "deep_core",
    },
    "innovation_drive": {
        "keywords": [
            "innovation", "transformative", "pilot", "new approach",
            "disrupt", "cutting-edge", "pioneer", "social innovation",
            "creative", "experiment", "prototype", "test bed",
            "entrepreneur", "startup", "scalable", "ambitious",
        ],
        "belief_level": "policy_core",
    },
    "austerity_scarcity": {
        "keywords": [
            "austerity", "cut", "budget reduction", "efficiency",
            "value for money", "savings", "fiscal", "reduced funding",
            "streamline", "rationalis", "cost-effective", "sustainable funding",
            "limited resource", "scarce", "insufficient", "funding gap",
            "economy", "economic", "budget", "funding",
        ],
        "belief_level": "policy_core",
    },
    "collaboration": {
        "keywords": [
            "partnership", "collaboration", "joint", "multi-agency",
            "cross-sector", "stakeholder engagement", "network",
            "collective", "coordinate", "integrated", "whole-system",
            "shared", "co-produc", "together", "alliance", "consortium",
        ],
        "belief_level": "secondary",
    },
    "evidence_based": {
        "keywords": [
            "evidence", "data", "research", "evaluation", "outcome",
            "impact measurement", "monitor", "indicator", "best practice",
            "what works", "proven", "pilot study", "longitudinal",
            "rigorous", "analysis", "baseline", "tracking",
        ],
        "belief_level": "secondary",
    },
    "cultural_identity": {
        "keywords": [
            "cultural", "tradition", "heritage", "language", "irish",
            "gaeltacht", "identity", "pride", "sense of place",
            "community spirit", "belonging", "roots", "local heritage",
            "storytelling", "music", "art", "culture",
        ],
        "belief_level": "deep_core",
    },
}

EMOTION_LEXICON = {
    "urgency": r"\b(crisis|urgent|immediate|critical|desperate|now|asap)\b",
    "frustration": r"\b(frustrat|disappoint|fed up|tired of|exhausting|"
                   r"bureaucracy|hopeless|impossible|barrier|blocking)\b",
    "hope": r"\b(hope|optimistic|promising|potential|opportunity|"
            r"encouraging|positive change|bright|future)\b",
    "fear": r"\b(worried|concerned|anxious|afraid|fear|threaten|"
            r"risk|danger|uncertain|precarious|fragile)\b",
}


# ── Helpers ───────────────────────────────────────────────────────────────

def safe_str(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    s = str(value).strip()
    return s if s.lower() not in {"", "nan", "none", "nat"} else default


def load_csv(paths: list[Path]) -> pd.DataFrame:
    for path in paths:
        if path.exists() and path.stat().st_size > 2:
            return pd.read_csv(path)
    return pd.DataFrame()


def load_nodes_edges():
    nodes = load_csv([ANALYTICS_DIR / "nodes.csv"])
    edges = load_csv([ANALYTICS_DIR / "edges.csv"])
    return nodes, edges


def text_for_node(row: pd.Series) -> str:
    for col in ["description", "quote", "label", "title"]:
        val = safe_str(row.get(col))
        if len(val) > 10:
            return val
    return safe_str(row.get("description", ""))


# ── Hyperbase Parser ──────────────────────────────────────────────────────

def init_parser():
    from hyperbase.parsers import get_parser
    return get_parser("alphabeta", params={"lang": "en"})


def parse_to_claim(text: str, parser) -> dict | None:
    """Parse a text through hyperbase and extract the main claim."""
    if not text or len(text.strip()) < 10:
        return None
    try:
        results = parser.parse(text)
    except Exception:
        return None
    if not results:
        return None
    result = results[0]
    if result.failed or result.errors:
        return None
    he = result.edge
    he_str = str(he)
    tokens = result.tokens

    verb, subject, obj = extract_svo(he)

    if not verb:
        return None

    return {
        "hyperedge": he_str,
        "verb": verb,
        "subject_raw": subject or "",
        "object_raw": obj or "",
        "tokens": " ".join(tokens) if tokens else "",
        "text": text,
    }


def extract_svo(he) -> tuple[str, str, str]:
    """Extract verb + first two meaningful concepts from a hyperbase hyperedge."""
    he_str = str(he)

    verb_match = re.search(r'(\w[\w-]*)/Pd\.\w+', he_str)
    verb = verb_match.group(1) if verb_match else ""

    atoms = re.findall(r'\(([^()]+?)/(?:Cc|Cp)\b', he_str)
    cleaned = []
    for a in atoms:
        a = re.sub(r'\s*/\w+\.?\w*', '', a).strip()
        a = re.sub(r'^(the|a|an|this|that|these|those|some|all|every)\s+', '', a, flags=re.IGNORECASE)
        a = re.sub(r'^(Md|Ma|Mn|Mi|Mm|Bx)\s+', '', a)
        a = a.strip()
        if a and len(a) > 1:
            cleaned.append(a)

    subject = cleaned[0] if len(cleaned) > 0 else ""
    obj = cleaned[1] if len(cleaned) > 1 else ""
    return verb, subject, obj


# ── Entity Linking ────────────────────────────────────────────────────────

def build_entity_index(nodes: pd.DataFrame) -> dict:
    """Build searchable index of operational entities for linking.

    Only indexes operational types (agent, project, pilot, prototype)
    to avoid linking claims back to the narrative nodes they came from.
    """
    OPERATIONAL_TYPES = {"agent", "project", "pilot", "prototype"}
    entities = {}
    spacy_available = False
    try:
        import spacy
        nlp = spacy.load("en_core_web_lg", disable=["lemmatizer"])
        spacy_available = True
    except Exception:
        nlp = None

    for _, row in nodes.iterrows():
        gid = safe_str(row.get("global_id"))
        label = safe_str(row.get("label"))
        ntype = safe_str(row.get("node_type"))
        if ntype not in OPERATIONAL_TYPES:
            continue
        desc = safe_str(row.get("description"))
        search_text = f"{label} {desc}"
        entities[gid] = {
            "global_id": gid,
            "label": label.lower(),
            "node_type": ntype,
            "search_text": search_text.lower(),
            "embedding": None,
        }

    # Precompute embeddings for entity matching
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [e["search_text"][:512] for e in entities.values()]
        if texts:
            emb = model.encode(texts, show_progress_bar=False)
            for i, eid in enumerate(entities):
                entities[eid]["embedding"] = emb[i]
    except Exception:
        pass

    return entities, nlp if spacy_available else None


def extract_entity_mentions(text: str, nlp) -> list[str]:
    """Extract noun chunks and named entities from text as candidate mentions."""
    mentions = []
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            mentions.append(ent.text.lower().strip())
        try:
            for chunk in doc.noun_chunks:
                text_c = chunk.text.lower().strip()
                if len(text_c) > 3 and text_c not in mentions:
                    mentions.append(text_c)
        except ValueError:
            pass
    # Also extract capitalized phrases as potential entity references
    cap_phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
    for p in cap_phrases:
        p_lower = p.lower()
        if p_lower not in mentions:
            mentions.append(p_lower)
    return list(set(mentions))


# Module-level cache for embedding model
_EMBEDDING_MODEL = None

def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


def link_entities(text: str, entity_index: dict, source_node_id: str,
                  edges: pd.DataFrame, nlp) -> dict:
    """Link entities mentioned in text to operational graph entities.

    Uses SentenceTransformer embedding similarity + token overlap.
    Only returns links with confidence >= 0.6.
    """
    mentions = extract_entity_mentions(text, nlp)
    if not mentions:
        return {"subject_entity_id": "", "object_entity_id": ""}

    model = _get_embedding_model()

    # Precompute mention embeddings in one batch
    mention_emb = {}
    if model is not None:
        try:
            embs = model.encode(mentions, show_progress_bar=False)
            for m, emb in zip(mentions, embs):
                mention_emb[m] = emb
        except Exception:
            pass

    import numpy as np

    # Score each mention against each entity
    mention_scores: dict[str, list[tuple[float, str, str]]] = {}
    for mention in mentions:
        mention_tokens = set(re.findall(r"[a-z]+", mention))
        scores = []
        for gid, info in entity_index.items():
            score = 0.0
            label_tokens = set(re.findall(r"[a-z]+", info["label"]))

            # Direct substring match
            if mention == info["label"]:
                score = 1.5
            elif mention in info["label"] or info["label"] in mention:
                score = 1.0
            elif len(mention) > 4 and len(info["label"]) > 4 and (
                mention.startswith(info["label"]) or info["label"].startswith(mention)
            ):
                score = 0.8

            # Token Jaccard overlap
            if mention_tokens and label_tokens:
                jac = len(mention_tokens & label_tokens) / max(len(mention_tokens | label_tokens), 1)
                score = max(score, jac * 1.2)

            # Embedding cosine similarity
            if info["embedding"] is not None and mention in mention_emb:
                try:
                    cos_sim = float(np.dot(mention_emb[mention], info["embedding"]) / (
                        np.linalg.norm(mention_emb[mention]) * np.linalg.norm(info["embedding"]) + 1e-10
                    ))
                    score = max(score, cos_sim * 0.5)
                except Exception:
                    pass

            # Graph context bonus
            if not edges.empty and source_node_id:
                mask = (
                    (edges["source_global_id"] == source_node_id) &
                    (edges["target_global_id"] == gid)
                ) | (
                    (edges["target_global_id"] == source_node_id) &
                    (edges["source_global_id"] == gid)
                )
                if mask.any():
                    score += 0.2

            if score > 0.3:
                scores.append((score, gid, info["label"]))

        if scores:
            scores.sort(key=lambda x: -x[0])
            mention_scores[mention] = scores

    if not mention_scores:
        return {"subject_entity_id": "", "object_entity_id": ""}

    # Pick the best overall entity match
    all_candidates = []
    for mention, scores in mention_scores.items():
        all_candidates.extend(scores)
    all_candidates.sort(key=lambda x: -x[0])

    best = all_candidates[0]
    second = all_candidates[1] if len(all_candidates) > 1 else None

    linked_subject = best[1] if best[0] >= 0.6 else ""
    linked_object = second[1] if second and second[0] >= 0.6 and second[1] != linked_subject else ""

    return {
        "subject_entity_id": linked_subject or "",
        "object_entity_id": linked_object or "",
    }


# ── Implicit Claim Detection ──────────────────────────────────────────────

def detect_negated(text: str) -> bool:
    return bool(NEGATION_MARKERS.search(text))


def detect_conditional(text: str) -> bool:
    return bool(CONDITIONAL_MARKERS.search(text))


def detect_emergency(text: str) -> bool:
    return bool(EMERGENCY_MARKERS.search(text))


def detect_emotion(text: str) -> dict:
    emotions = {}
    for label, pattern in EMOTION_LEXICON.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            emotions[label] = len(matches)
    return emotions


def infer_implicit_claim(surface: dict, text: str) -> dict | None:
    """Given a surface claim, infer what implicit claim it suggests."""
    is_negated = detect_negated(text)
    is_conditional = detect_conditional(text)
    is_emergency = detect_emergency(text)
    emotions = detect_emotion(text)

    if not (is_negated or is_conditional or is_emergency or emotions):
        return None

    implied = dict(surface)
    implied["narrative_level"] = "implicit"
    implied["negated"] = is_negated
    implied["conditional"] = is_conditional
    implied["emergency_frame"] = is_emergency
    implied["emotions"] = json.dumps(emotions)

    # Transform the claim for implicit version
    verb = surface.get("verb", "")
    subj = surface.get("subject_raw", "")
    obj = surface.get("object_raw", "")

    if is_negated:
        # "We don't have funding" → implies "funding is missing"
        implied["verb"] = "lacks"
        implied["subject_raw"] = subj or "system"
        implied["object_raw"] = obj or verb
        implied["inference_rule"] = "negation_inversion"
    elif is_emergency:
        # "Youth services are at risk" → implies "youth services need protection"
        implied["verb"] = "needs_protection"
        implied["subject_raw"] = subj or "system"
        implied["object_raw"] = obj or "stability"
        implied["inference_rule"] = "emergency_frame"
    elif is_conditional:
        implied["verb"] = "contingent_on"
        implied["inference_rule"] = "conditional_dependency"

    return implied


# ── Metanarrative Classification ──────────────────────────────────────────

def classify_metanarrative(text: str) -> list[dict]:
    """Classify text into value dimensions / metanarratives."""
    matches = []
    text_lower = text.lower()
    for dimension, config in VALUE_DIMENSIONS.items():
        score = 0
        matched_kws = []
        for kw in config["keywords"]:
            if kw.lower() in text_lower:
                score += 1
                matched_kws.append(kw)
        if score > 0:
            matches.append({
                "value_dimension": dimension,
                "belief_level": config["belief_level"],
                "score": score,
                "matched_keywords": matched_kws,
            })
    matches.sort(key=lambda x: -x["score"])
    return matches


# ── Main Pipeline ─────────────────────────────────────────────────────────

def main() -> None:
    print(f"Narrative Layer Extraction — {PLATFORM_ID}/{OUTPUT_SUBDIR}")
    print("=" * 50)

    nodes, edges = load_nodes_edges()
    if nodes.empty:
        print("ERROR: No nodes found.")
        return

    print(f"Loaded {len(nodes)} nodes, {len(edges) if not edges.empty else 0} edges")

    # 1. Select narrative nodes
    NARRATIVE_TYPES = {"information"}
    narrative_nodes = nodes[nodes["node_type"].isin(NARRATIVE_TYPES)].copy()
    narrative_nodes = narrative_nodes[narrative_nodes.apply(
        lambda r: len(text_for_node(r)) > 10, axis=1
    )]
    print(f"Narrative nodes with text: {len(narrative_nodes)}")

    if narrative_nodes.empty:
        print("No narrative text found. Nothing to extract.")
        return

    # 2. Build entity index for linking
    entity_index, nlp = build_entity_index(nodes)
    print(f"Entity index: {len(entity_index)} entries")

    # 3. Initialize hyperbase parser
    print("Initializing hyperbase alphabeta parser...")
    try:
        parser = init_parser()
        print("Parser ready.")
    except Exception as e:
        print(f"ERROR: Could not initialize parser: {e}")
        print("Is hyperbase + hyperbase-parser-ab installed?")
        print("  pip install hyperbase hyperbase-parser-ab")
        return

    # 4. Extract claims
    all_surface = []
    all_implicit = []
    all_metanarratives = []
    claim_idx = 1

    for _, row in narrative_nodes.iterrows():
        text = text_for_node(row)
        gid = safe_str(row["global_id"])
        ntype = safe_str(row["node_type"])

        # Parse surface claim
        claim = parse_to_claim(text, parser)
        if not claim:
            continue

        claim["claim_id"] = f"claim_{claim_idx}"
        claim["source_node_id"] = gid
        claim["source_node_type"] = ntype
        claim["narrative_level"] = "surface"
        claim_idx += 1

        # Entity linking — uses original text + spaCy NER + embedding similarity
        links = link_entities(text, entity_index, gid, edges, nlp)
        claim.update(links)

        # Metaphor detection + implicit inference
        implicit = infer_implicit_claim(claim, text)
        if implicit:
            implicit["claim_id"] = f"claim_{claim_idx}"
            implicit["source_node_id"] = gid
            claim_idx += 1
            # Re-link for implicit claim
            implicit_links = link_entities(implicit.get("text", text), entity_index, gid, edges, nlp)
            implicit.update(implicit_links)
            all_implicit.append(implicit)

        # Metanarrative classification
        meta = classify_metanarrative(text)
        if meta:
            for m in meta:
                m["claim_id"] = claim["claim_id"]
                m["source_node_id"] = gid
                m["source_text"] = text[:200]
            all_metanarratives.extend(meta)

        all_surface.append(claim)

    print(f"Surface claims: {len(all_surface)}")
    print(f"Implicit claims: {len(all_implicit)}")
    print(f"Metanarrative classifications: {len(all_metanarratives)}")

    # 5. Build output DataFrames
    surface_df = pd.DataFrame(all_surface) if all_surface else pd.DataFrame()
    implicit_df = pd.DataFrame(all_implicit) if all_implicit else pd.DataFrame()
    meta_df = pd.DataFrame(all_metanarratives) if all_metanarratives else pd.DataFrame()

    # Claim nodes — one row per unique claim ID
    claim_nodes = []
    seen_ids = set()
    for claim in all_surface + all_implicit:
        cid = claim.get("claim_id")
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        claim_nodes.append({
            "global_id": cid,
            "node_type": "claim",
            "label": f"{claim.get('verb', '?')}: {claim.get('subject_raw', '')} → {claim.get('object_raw', '')}",
            "description": claim.get("text", "")[:500],
            "narrative_level": claim.get("narrative_level", "surface"),
            "verb": claim.get("verb", ""),
            "subject_raw": claim.get("subject_raw", ""),
            "object_raw": claim.get("object_raw", ""),
            "hyperedge": claim.get("hyperedge", ""),
            "subject_entity_id": claim.get("subject_entity_id", ""),
            "object_entity_id": claim.get("object_entity_id", ""),
            "source_node_id": claim.get("source_node_id", ""),
            "source_node_type": claim.get("source_node_type", ""),
            "negated": str(claim.get("negated", False)),
            "conditional": str(claim.get("conditional", False)),
            "emergency_frame": str(claim.get("emergency_frame", False)),
            "emotions": claim.get("emotions", ""),
            "inference_rule": claim.get("inference_rule", ""),
            "platform_id": PLATFORM_ID,
        })

    claim_df = pd.DataFrame(claim_nodes)
    # Assign value dimensions + belief levels
    if not meta_df.empty:
        dim_map = meta_df.groupby("claim_id").apply(
            lambda g: g.sort_values("score", ascending=False).iloc[0]
        ).reset_index(drop=True)
        dim_lookup = dict(zip(dim_map["claim_id"], dim_map["value_dimension"]))
        belief_lookup = dict(zip(dim_map["claim_id"], dim_map["belief_level"]))
        claim_df["value_dimension"] = claim_df["global_id"].map(dim_lookup).fillna("")
        claim_df["belief_level"] = claim_df["global_id"].map(belief_lookup).fillna("")
    else:
        claim_df["value_dimension"] = ""
        claim_df["belief_level"] = ""

    # Claim edges: source_node → claim, claim → subject_entity, claim → object_entity
    claim_edges = []
    for _, cr in claim_df.iterrows():
        cid = cr["global_id"]
        src = cr["source_node_id"]
        if src:
            claim_edges.append({
                "edge_id": f"e_{cid}_from_{src}",
                "source_global_id": src,
                "target_global_id": cid,
                "edge_type": f"{cr['source_node_type']}_makes_claim",
                "edge_family": "narrative_claim",
                "directed": True,
                "weight": 1.0,
                "methodological_phase": "narrative_extraction",
            })
        subj_id = cr["subject_entity_id"]
        if subj_id and subj_id != src:
            claim_edges.append({
                "edge_id": f"e_{cid}_subj_{subj_id}",
                "source_global_id": cid,
                "target_global_id": subj_id,
                "edge_type": "claim_about_subject",
                "edge_family": "narrative_claim",
                "directed": True,
                "weight": 0.8,
                "methodological_phase": "narrative_extraction",
            })
        obj_id = cr["object_entity_id"]
        if obj_id and obj_id != src and obj_id != subj_id:
            claim_edges.append({
                "edge_id": f"e_{cid}_obj_{obj_id}",
                "source_global_id": cid,
                "target_global_id": obj_id,
                "edge_type": "claim_about_object",
                "edge_family": "narrative_claim",
                "directed": True,
                "weight": 0.8,
                "methodological_phase": "narrative_extraction",
            })

    claim_edge_df = pd.DataFrame(claim_edges)

    # 6. Write outputs
    out_dir = ANALYSIS_DIR / "narrative_layers"
    out_dir.mkdir(parents=True, exist_ok=True)

    claim_df.to_csv(out_dir / "claim_nodes.csv", index=False)
    claim_edge_df.to_csv(out_dir / "claim_edges.csv", index=False)
    if not surface_df.empty:
        surface_df.to_csv(out_dir / "surface_claims.csv", index=False)
    if not implicit_df.empty:
        implicit_df.to_csv(out_dir / "implicit_claims.csv", index=False)
    if not meta_df.empty:
        meta_df.to_csv(out_dir / "metanarratives.csv", index=False)

    # Summary
    level_counts = claim_df["narrative_level"].value_counts().to_dict()
    dim_counts = {}
    for dim in VALUE_DIMENSIONS:
        count = int((claim_df["value_dimension"] == dim).sum())
        if count:
            dim_counts[dim] = count

    summary = {
        "narrative_nodes_processed": len(narrative_nodes),
        "total_claims": len(claim_df),
        "surface_claims": level_counts.get("surface", 0),
        "implicit_claims": level_counts.get("implicit", 0),
        "claims_with_subject_link": int(claim_df["subject_entity_id"].astype(str).str.strip().ne("").sum()),
        "claims_with_object_link": int(claim_df["object_entity_id"].astype(str).str.strip().ne("").sum()),
        "total_claim_edges": len(claim_edge_df),
        "value_dimensions_found": dim_counts,
        "metanarrative_classifications": len(meta_df),
    }

    out_summary = out_dir / "narrative_extraction_summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    print(f"\nOutput: {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()

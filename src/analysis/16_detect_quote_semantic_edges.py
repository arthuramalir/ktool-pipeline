from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173_synthetic")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
BASE_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR
ANALYSIS_DIR = BASE_DIR / "analysis"
REL_DIR = BASE_DIR / "relationships"

CAUSAL_MARKERS = re.compile(r"\b(because|therefore|so that|due to|because of|resulted in|led to|creates?|enables?|causes?)\b", re.IGNORECASE)
CONTRADICTION_MARKERS = re.compile(r"\b(but|however|although|yet|instead|despite|problem|barrier|conflict|risk|gap)\b", re.IGNORECASE)


def safe_text(*values: object) -> str:
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none"}:
            return text
    return ""


def load_quotes() -> pd.DataFrame:
    candidates = [ANALYSIS_DIR / "quote_candidates.csv", REL_DIR / "quote_candidates.csv"]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    return pd.DataFrame()


def text_for_similarity(df: pd.DataFrame) -> list[str]:
    return [
        safe_text(row.get("quote"), row.get("description"), row.get("node_label"), row.get("information_id"))
        for _, row in df.iterrows()
    ]


def relation_rows(quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame()

    texts = text_for_similarity(quotes)
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts) if texts else None
    similarity = cosine_similarity(matrix) if matrix is not None and len(texts) > 1 else np.zeros((len(texts), len(texts)))

    rows = []
    channel_groups = {}
    for idx, row in quotes.iterrows():
        channel = safe_text(row.get("channel_id"), row.get("channel_code")) or "no_channel"
        channel_groups.setdefault(channel, []).append(idx)

    # Within-channel sequences.
    for channel, indices in channel_groups.items():
        ordered = sorted(indices, key=lambda ix: safe_text(quotes.iloc[ix].get("information_id")))
        for left, right in zip(ordered, ordered[1:]):
            source = quotes.iloc[left]
            target = quotes.iloc[right]
            rows.append(
                {
                    "source_global_id": safe_text(source.get("information_id")),
                    "target_global_id": safe_text(target.get("information_id")),
                    "edge_type": "sequence",
                    "edge_family": "qualitative_narrative",
                    "methodological_phase": "listening",
                    "directed": True,
                    "evidence_source": f"channel_sequence:{channel}",
                    "source_quote": safe_text(source.get("quote")),
                    "target_quote": safe_text(target.get("quote")),
                    "weight": 0.85,
                    "parameter": "Sequential narrative flow",
                    "narrative_coalition": "Potential",
                    "description": f"Sequential relation in channel {channel}",
                }
            )

    # Cross-channel semantic relations.
    for i in range(len(quotes)):
        for j in range(i + 1, len(quotes)):
            src = quotes.iloc[i]
            tgt = quotes.iloc[j]
            if safe_text(src.get("channel_id")) == safe_text(tgt.get("channel_id")):
                continue
            sim = float(similarity[i, j]) if similarity is not None else 0.0
            themes_src = set(safe_text(src.get("topics_sub_areas"), src.get("topics_thematic_areas")).replace("|", ",").split(","))
            themes_tgt = set(safe_text(tgt.get("topics_sub_areas"), tgt.get("topics_thematic_areas")).replace("|", ",").split(","))
            shared_themes = {item.strip() for item in themes_src.intersection(themes_tgt) if item.strip()}
            text_src = safe_text(src.get("quote"), src.get("description"))
            text_tgt = safe_text(tgt.get("quote"), tgt.get("description"))

            if not shared_themes and sim < 0.15:
                continue

            relation_type = "similarity"
            weight = 0.5 + 0.5 * min(sim, 1.0)
            parameter = "Thematic similarity"
            description = f"Quotes align around {', '.join(sorted(shared_themes)) if shared_themes else 'shared language'}"
            if CAUSAL_MARKERS.search(text_src) or CAUSAL_MARKERS.search(text_tgt):
                relation_type = "causality"
                weight = max(weight, 0.7)
                parameter = "Cause-and-effect language"
                description = "One or both quotes use causal language to frame the relationship."
            elif CONTRADICTION_MARKERS.search(text_src) or CONTRADICTION_MARKERS.search(text_tgt):
                relation_type = "contradiction"
                weight = max(weight, 0.65)
                parameter = "Contrasting language"
                description = "One or both quotes include tension or opposition language."
            elif sim < 0.3 and shared_themes:
                relation_type = "frequency"
                weight = 0.6
                parameter = "Repeated theme"
                description = f"Repeated theme across channels: {', '.join(sorted(shared_themes))}"

            rows.append(
                {
                    "source_global_id": safe_text(src.get("information_id")),
                    "target_global_id": safe_text(tgt.get("information_id")),
                    "edge_type": relation_type,
                    "edge_family": "qualitative_narrative",
                    "methodological_phase": "listening",
                    "directed": False,
                    "evidence_source": "tfidf_quote_semantic_detector",
                    "source_quote": text_src,
                    "target_quote": text_tgt,
                    "weight": round(float(weight), 4),
                    "parameter": parameter,
                    "narrative_coalition": "Yes" if relation_type in {"similarity", "frequency"} else "Potential",
                    "description": description,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    quotes = load_quotes()
    if quotes.empty:
        print("[ERROR] Quote candidates not found. Run 15_extract_relevant_quotes.py first.")
        return

    edges = relation_rows(quotes)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REL_DIR.mkdir(parents=True, exist_ok=True)
    if not edges.empty:
        edges.to_csv(REL_DIR / "quote_semantic_edges.csv", index=False)
        edges.to_csv(ANALYSIS_DIR / "quote_semantic_edges.csv", index=False)
        print(f"Wrote {len(edges)} quote semantic edges.")
    else:
        print("[WARNING] No quote semantic edges were produced.")


if __name__ == "__main__":
    main()

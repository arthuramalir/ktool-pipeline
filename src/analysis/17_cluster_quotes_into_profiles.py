from __future__ import annotations

import os
from pathlib import Path

import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173_synthetic")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
BASE_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR
ANALYSIS_DIR = BASE_DIR / "analysis"
REL_DIR = BASE_DIR / "relationships"


def safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return text


def load_quotes() -> pd.DataFrame:
    for path in [ANALYSIS_DIR / "quote_candidates.csv", REL_DIR / "quote_candidates.csv"]:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    return pd.DataFrame()


def load_edges() -> pd.DataFrame:
    for path in [REL_DIR / "quote_semantic_edges.csv", ANALYSIS_DIR / "quote_semantic_edges.csv"]:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    return pd.DataFrame()


def make_cluster_label(frame: pd.DataFrame) -> str:
    themes = []
    for column in ["topics_thematic_areas", "topics_sub_areas", "values"]:
        for text in frame.get(column, pd.Series(dtype=str)).fillna("").astype(str).tolist():
            for part in text.replace("|", ",").split(","):
                part = part.strip()
                if part:
                    themes.append(part)
    if themes:
        counts = pd.Series(themes).value_counts()
        return str(counts.index[0])
    return "mixed narrative"


def cluster_quotes(quotes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame()

    texts = (
        quotes.get("quote", pd.Series(index=quotes.index, dtype=str)).fillna("").astype(str)
        + " "
        + quotes.get("description", pd.Series(index=quotes.index, dtype=str)).fillna("").astype(str)
    ).tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts) if texts else None

    if matrix is not None and len(quotes) > 2:
        cluster_count = max(2, min(8, len(quotes) // 4 + 1))
        model = AgglomerativeClustering(n_clusters=cluster_count)
        cluster_ids = model.fit_predict(matrix.toarray())
    else:
        cluster_ids = [0] * len(quotes)

    graph = nx.Graph()
    for _, row in edges.iterrows():
        if float(row.get("weight", 0) or 0) < 0.55:
            continue
        graph.add_edge(safe_text(row.get("source_global_id")), safe_text(row.get("target_global_id")), weight=float(row.get("weight", 1.0) or 1.0))

    quote_cluster_lookup = {safe_text(quotes.iloc[i].get("quote_id")): int(cluster_ids[i]) for i in range(len(quotes))}

    rows = []
    for quote_idx, row in quotes.iterrows():
        quote_id = safe_text(row.get("quote_id"))
        cluster_id = quote_cluster_lookup.get(quote_id, 0)
        rows.append(
            {
                "quote_id": quote_id,
                "information_id": safe_text(row.get("information_id")),
                "cluster_id": f"cluster_{cluster_id + 1}",
                "quote": safe_text(row.get("quote")),
                "channel_id": safe_text(row.get("channel_id")),
                "channel_code": safe_text(row.get("channel_code")),
                "selection_score": float(row.get("selection_score", 0) or 0),
                "topics_thematic_areas": safe_text(row.get("topics_thematic_areas")),
                "topics_sub_areas": safe_text(row.get("topics_sub_areas")),
                "values": safe_text(row.get("values")),
            }
        )

    cluster_df = pd.DataFrame(rows)
    if cluster_df.empty:
        return cluster_df

    profiles = []
    for cluster_id, group in cluster_df.groupby("cluster_id"):
        profiles.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": make_cluster_label(group),
                "quote_count": int(len(group)),
                "information_count": int(group["information_id"].nunique()),
                "top_channels": " | ".join(group["channel_code"].fillna("").astype(str).value_counts().head(3).index.tolist()),
                "top_themes": make_cluster_label(group),
                "average_selection_score": round(float(group["selection_score"].mean()), 4),
                "representative_quote": safe_text(group.sort_values("selection_score", ascending=False).iloc[0].get("quote")),
            }
        )

    profile_df = pd.DataFrame(profiles).sort_values(["quote_count", "average_selection_score"], ascending=[False, False])
    return cluster_df, profile_df


def main() -> None:
    quotes = load_quotes()
    edges = load_edges()
    if quotes.empty:
        print("[ERROR] Quote candidates not found. Run 15_extract_relevant_quotes.py first.")
        return

    cluster_df, profile_df = cluster_quotes(quotes, edges)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    cluster_df.to_csv(ANALYSIS_DIR / "quote_clusters.csv", index=False)
    profile_df.to_csv(ANALYSIS_DIR / "narrative_profiles.csv", index=False)
    print(f"Wrote {len(cluster_df)} quote clusters and {len(profile_df)} narrative profiles.")


if __name__ == "__main__":
    main()

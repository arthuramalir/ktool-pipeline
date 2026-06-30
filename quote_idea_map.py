from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import umap

from strapi_utils import data_dir, project_root


HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")

# Expanded stop words to prevent cross-lingual cluster summaries from being overwhelmed by syntax particles
MULTILINGUAL_STOP_WORDS = [
    # English
    "the", "and", "of", "to", "a", "in", "is", "that", "it", "for", "on", "with", "as", "at", "this", "but", "by", "from",
    # Spanish
    "de", "la", "que", "el", "en", "y", "los", "un", "una", "con", "para", "por", "es", "una", "del", "al", "lo", "su", "sus", "como", "más", "pero",
    # Basque
    "eta", "da", "dira", "zen", "ziren", "du", "dute", "ere", "baino", "ez", "bai", "naiz", "gaitu", "ta", "hori", "hau", "dago", "dute"
]


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\u00a0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def text_excerpt(text: str, limit: int = 220) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def load_documents(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"id", "entity_type", "text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Documents are missing required columns: {sorted(missing)}")
    return frame


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)


def quote_mask(documents: pd.DataFrame) -> pd.Series:
    entity_type = documents["entity_type"].astype(str)
    return entity_type.str.contains(r"quote|transcription|transcript", case=False, regex=True)

def build_quote_map(documents: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    mask = quote_mask(documents)
    quote_docs = documents.loc[mask, ["id", "entity_type", "text"]].copy()
    quote_embeddings = embeddings[mask.to_numpy()]

    if len(quote_docs) == 0:
        raise ValueError("No transcription/quote rows were found in documents.parquet")

    # 1. FIX THE GHOSTS: Clean text first, then filter out empty or near-empty fragments
    quote_docs["clean_text"] = [clean_text(text) for text in quote_docs["text"].tolist()]
    
    # Only preserve rows that have actual substance (more than 10 characters)
    substantive_mask = quote_docs["clean_text"].str.strip().str.len() > 10
    if not substantive_mask.any():
        raise ValueError("No rows survived after filtering out ghost rows/empty texts.")
        
    quote_docs = quote_docs[substantive_mask].copy()
    quote_embeddings = quote_embeddings[substantive_mask.to_numpy()]
    
    # Generate excerpts on clean rows only
    quote_docs["text_excerpt"] = [text_excerpt(text) for text in quote_docs["clean_text"].tolist()]

    # 2. SHATTER THE MEGA-CLUSTER: Force UMAP and HDBSCAN to zoom in on local density
    proximity_space = normalize(np.asarray(quote_embeddings, dtype=np.float32))
    umap_coords = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,    # Focus heavily on immediate semantic neighbors
        min_dist=0.01,     # Pack tight ideological groupings closely together
    ).fit_transform(proximity_space)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15,  # Capture smaller, highly-specific perspective pockets
        min_samples=5,        # Prevent too many points from being classified as noise (-1)
        prediction_data=True,
    )
    cluster_labels = clusterer.fit_predict(proximity_space)
    cluster_probabilities = getattr(clusterer, "probabilities_", np.ones(len(quote_docs), dtype=np.float32))

    frame = quote_docs.assign(
        x=umap_coords[:, 0],
        y=umap_coords[:, 1],
        cluster_id=cluster_labels.astype(int),
        cluster_probability=np.asarray(cluster_probabilities, dtype=np.float32),
    )
    return frame

def cluster_summary(frame: pd.DataFrame) -> pd.DataFrame:
    # UPDATED: Added stop_words parameter to isolate meaningful vocabulary concepts
    quote_terms = TfidfVectorizer(
        stop_words=MULTILINGUAL_STOP_WORDS,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        token_pattern=r"(?u)\b[\w\-]{3,}\b",
    )
    tfidf = quote_terms.fit_transform(frame["clean_text"].tolist())
    terms = np.asarray(quote_terms.get_feature_names_out())

    rows: list[dict[str, object]] = []
    for cluster_id, subset in frame.groupby("cluster_id"):
        member_idx = subset.index.to_list()
        if not member_idx:
            continue
        mean_vec = np.asarray(tfidf[frame.index.isin(member_idx)].mean(axis=0)).ravel()
        
        if len(mean_vec) == 0:
            top_terms = ""
        else:
            top_idx = mean_vec.argsort()[-8:][::-1]
            top_terms = ", ".join(terms[top_idx].tolist())
            
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "count": int(len(subset)),
                "mean_probability": float(subset["cluster_probability"].mean()),
                "top_terms": top_terms,
            }
        )
    return pd.DataFrame(rows).sort_values(["cluster_id"]).reset_index(drop=True)


def save_figure(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = px.scatter(
        frame,
        x="x",
        y="y",
        color=frame["cluster_id"].astype(str),
        hover_name="id",
        hover_data={
            "entity_type": True,
            "text_excerpt": True,
            "cluster_probability": ":.3f",
            "x": False,
            "y": False,
            "cluster_id": False,
            "clean_text": False,
        },
        title="Quote idea proximity map",
        template="plotly_white",
        width=1200,
        height=850,
    )
    fig.update_traces(marker=dict(size=6, opacity=0.75, line=dict(width=0.4, color="white")))
    fig.update_layout(legend_title_text="Idea cluster", hoverlabel=dict(bgcolor="white", font_size=12))
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a quote-centric semantic idea map")
    parser.add_argument("--documents", type=Path, default=data_dir() / "cleaned_transcript_documents.parquet")
    parser.add_argument("--embeddings", type=Path, default=data_dir() / "cleaned_embeddings_sentence_transformer.npy")
    parser.add_argument("--output", type=Path, default=data_dir() / "cleaned_quote_idea_map.parquet")
    parser.add_argument("--summary-output", type=Path, default=data_dir() / "cleaned_quote_idea_cluster_summary.parquet")
    parser.add_argument("--figure", type=Path, default=project_root() / "figures" / "cleaned_quote_idea_map.html")
    args = parser.parse_args()

    documents = load_documents(args.documents)
    embeddings = load_embeddings(args.embeddings)
    if len(documents) != len(embeddings):
        raise ValueError("Document count and embedding count do not match")

    frame = build_quote_map(documents, embeddings)
    summary = cluster_summary(frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    summary.to_parquet(args.summary_output, index=False)
    save_figure(frame, args.figure)

    print(f"Wrote quote idea map to {args.output}")
    print(f"Wrote quote cluster summary to {args.summary_output}")
    print(f"Wrote quote idea figure to {args.figure}")


if __name__ == "__main__":
    main()
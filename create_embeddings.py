from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from strapi_utils import data_dir

# A basic combined list of common stop words across English and Spanish.
# Note: Basque is agglutinative; true Basque stop-word filtering is usually done via lemmatization, 
# but adding common pronouns/connectors here helps keep the vocabulary cleaner.
MULTILINGUAL_STOP_WORDS = [
    # English
    "the", "a", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "is", "are", "was", "were",
    # Spanish
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero", "en", "para", "por", "de", "con", "es", "son", "fue", "eran", "que", "en", "su", "sus",
    # Basque (Common particles/connectors)
    "eta", "da", "dira", "zen", "ziren", "du", "dute", "eta", "ere", "baino", "ez", "bai", "naiz", "gaitu"
]

def load_documents(documents_path: Path) -> pd.DataFrame:
    if documents_path.suffix == ".parquet":
        return pq.read_table(documents_path).to_pandas()
    with documents_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create semantic and lexical document representations")
    parser.add_argument("--documents", type=Path, default=data_dir() / "cleaned_transcript_documents.parquet")
    parser.add_argument("--sentence-embeddings", type=Path, default=data_dir() / "cleaned_embeddings_sentence_transformer.npy")
    parser.add_argument("--tfidf", type=Path, default=data_dir() / "cleaned_tfidf_matrix.npz")
    parser.add_argument("--tfidf-vocab", type=Path, default=data_dir() / "cleaned_tfidf_vocabulary.json")
    
    # SWAPPED: Changed from 'all-MiniLM-L6-v2' to a highly aligned cross-lingual model
    parser.add_argument("--model", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    args = parser.parse_args()

    documents = load_documents(args.documents)
    texts = documents["text"].fillna("").astype(str).tolist()

    print(f"Loading cross-lingual embedding model: {args.model}...")
    model = SentenceTransformer(args.model)
    sentence_embeddings = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
    np.save(args.sentence_embeddings, sentence_embeddings)

    # UPDATED: Swapped to multilingual stop words to prevent English rules from scrambling things
    vectorizer = TfidfVectorizer(
        stop_words=MULTILINGUAL_STOP_WORDS,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    sparse.save_npz(args.tfidf, tfidf_matrix)
    tfidf_vocabulary = {term: int(index) for term, index in vectorizer.vocabulary_.items()}
    args.tfidf_vocab.write_text(json.dumps(tfidf_vocabulary, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")

    print(f"Saved sentence-transformer embeddings to {args.sentence_embeddings}")
    print(f"Saved TF-IDF matrix to {args.tfidf}")


if __name__ == "__main__":
    main()
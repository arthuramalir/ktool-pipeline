from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
from scipy.sparse import load_npz

from strapi_utils import data_dir


def top_keywords(texts: list[str], model: KeyBERT, top_n: int = 8) -> list[str]:
    corpus = "\n".join(texts[:50])
    keywords = model.extract_keywords(corpus, top_n=top_n, stop_words="english", use_mmr=True, diversity=0.4)
    return [kw for kw, _ in keywords]


def load_tfidf_terms(tfidf_path: Path, vocab_path: Path) -> tuple[np.ndarray, list[str]]:
    tfidf = load_npz(tfidf_path)
    vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
    terms = [term for term, idx in sorted(vocabulary.items(), key=lambda item: item[1])]
    return tfidf, terms


def cluster_keywords(tfidf_matrix, doc_indices: list[int], terms: list[str], top_n: int = 8) -> list[dict[str, Any]]:
    if not doc_indices:
        return []
    matrix = tfidf_matrix[doc_indices]
    weights = np.asarray(matrix.mean(axis=0)).ravel()
    top_idx = np.argsort(weights)[::-1][:top_n]
    return [{"term": terms[i], "importance": float(weights[i])} for i in top_idx if weights[i] > 0]


def graph_degree_map(graph_path: Path) -> dict[str, float]:
    if not graph_path.exists():
        return {}
    import networkx as nx

    graph = nx.read_graphml(graph_path)
    undirected = graph.to_undirected()
    degree = nx.degree_centrality(undirected)
    return {str(node): float(score) for node, score in degree.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain clusters using keywords and representative documents")
    parser.add_argument("--documents", type=Path, default=data_dir() / "documents.parquet")
    parser.add_argument("--consensus", type=Path, default=data_dir() / "consensus_perception_space.parquet")
    parser.add_argument("--embeddings", type=Path, default=data_dir() / "embeddings_sentence_transformer.npy")
    parser.add_argument("--tfidf", type=Path, default=data_dir() / "tfidf_matrix.npz")
    parser.add_argument("--tfidf-vocabulary", type=Path, default=data_dir() / "tfidf_vocabulary.json")
    parser.add_argument("--graph", type=Path, default=data_dir() / "graph_enriched.graphml")
    parser.add_argument("--output", type=Path, default=data_dir() / "cluster_explanations.json")
    parser.add_argument("--doc-output", type=Path, default=data_dir() / "document_explanations.parquet")
    args = parser.parse_args()

    documents = pd.read_parquet(args.documents)
    consensus = pd.read_parquet(args.consensus)
    embeddings = np.load(args.embeddings)
    tfidf_matrix, terms = load_tfidf_terms(args.tfidf, args.tfidf_vocabulary)
    text_map = dict(zip(documents["id"].astype(str), documents["text"].astype(str)))
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(documents["id"].astype(str).tolist())}
    kw_model = KeyBERT(model=SentenceTransformer("all-MiniLM-L6-v2"))
    graph_degree = graph_degree_map(args.graph)

    explanations = []
    document_rows = []
    cluster_members = consensus.groupby("cluster_consensus_id")
    for cluster_id, members in cluster_members:
        member_ids = members["document_id"].astype(str).tolist()
        member_indices = [id_to_idx[d] for d in member_ids if d in id_to_idx]
        if not member_indices:
            continue

        cluster_texts = [text_map.get(doc_id, "") for doc_id in member_ids if text_map.get(doc_id)]
        keywords = top_keywords(cluster_texts, kw_model)
        evidence = cluster_keywords(tfidf_matrix, member_indices, terms)
        centroid = embeddings[member_indices].mean(axis=0)
        centroid_vec = centroid.reshape(1, -1)
        similarities = cosine_similarity(embeddings, centroid_vec).ravel()
        semantic_scores = pd.Series(similarities, index=documents["id"].astype(str))

        typical_docs = (
            members.sort_values(["cluster_confidence", "entropy_score"], ascending=[False, True])["document_id"].astype(str).tolist()[:3]
        )
        connected_docs = sorted(
            member_ids,
            key=lambda doc_id: graph_degree.get(doc_id, 0.0),
            reverse=True,
        )[:3]

        boundary_candidates = []
        for doc_id in member_ids:
            idx = id_to_idx.get(doc_id)
            if idx is None:
                continue
            vector = embeddings[idx].reshape(1, -1)
            similarities_to_clusters = []
            for other_cluster, other_members in cluster_members:
                other_indices = [id_to_idx[d] for d in other_members["document_id"].astype(str).tolist() if d in id_to_idx]
                if not other_indices:
                    continue
                other_centroid = embeddings[other_indices].mean(axis=0).reshape(1, -1)
                similarities_to_clusters.append((int(other_cluster), float(cosine_similarity(vector, other_centroid)[0, 0])))
            similarities_to_clusters.sort(key=lambda item: item[1], reverse=True)
            if len(similarities_to_clusters) >= 2:
                margin = similarities_to_clusters[0][1] - similarities_to_clusters[1][1]
                boundary_candidates.append(
                    {
                        "document_id": doc_id,
                        "best_cluster": similarities_to_clusters[0][0],
                        "runner_up_cluster": similarities_to_clusters[1][0],
                        "margin": float(margin),
                    }
                )

        boundary_candidates = sorted(boundary_candidates, key=lambda item: item["margin"])[:3]
        label = " / ".join(keywords[:3]) if keywords else f"cluster_{cluster_id}"
        explanations.append(
            {
                "cluster": int(cluster_id),
                "label": label,
                "keywords": evidence or [{"term": kw, "importance": 1.0} for kw in keywords],
                "representative_documents": {
                    "closest_to_centroid": [doc for doc, _ in semantic_scores.loc[member_ids].sort_values(ascending=False).head(3).items()],
                    "most_connected": connected_docs,
                    "most_typical": typical_docs,
                },
                "boundary_examples": boundary_candidates,
                "confidence": float(members["cluster_confidence"].mean()),
            }
        )

        cluster_probabilities = members.set_index("document_id")["cluster_confidence"].to_dict()
        for doc_id in member_ids:
            idx = id_to_idx.get(doc_id)
            if idx is None:
                continue
            vector = embeddings[idx].reshape(1, -1)
            cluster_similarity_rows = []
            for other_cluster, other_members in cluster_members:
                other_indices = [id_to_idx[d] for d in other_members["document_id"].astype(str).tolist() if d in id_to_idx]
                if not other_indices:
                    continue
                other_centroid = embeddings[other_indices].mean(axis=0).reshape(1, -1)
                similarity = float(cosine_similarity(vector, other_centroid)[0, 0])
                cluster_similarity_rows.append((int(other_cluster), similarity))

            cluster_similarity_rows.sort(key=lambda item: item[1], reverse=True)
            similarities = np.asarray([score for _, score in cluster_similarity_rows], dtype=np.float32)
            probs = softmax(similarities) if len(similarities) else np.asarray([1.0], dtype=np.float32)
            alt_index = 1 if len(cluster_similarity_rows) > 1 else 0
            document_rows.append(
                {
                    "document_id": doc_id,
                    "primary_cluster": int(cluster_id),
                    "primary_label": label,
                    "primary_confidence": float(probs[0] if len(probs) else 1.0),
                    "alternative_cluster": int(cluster_similarity_rows[alt_index][0]) if len(cluster_similarity_rows) > 1 else int(cluster_id),
                    "alternative_confidence": float(probs[alt_index] if len(probs) > alt_index else 0.0),
                    "ambiguity_score": float(1.0 - (probs[0] if len(probs) else 1.0)),
                    "nearest_cluster_margin": float(cluster_similarity_rows[0][1] - cluster_similarity_rows[1][1]) if len(cluster_similarity_rows) > 1 else 0.0,
                    "nearest_representatives": json.dumps(explanations[-1]["representative_documents"], ensure_ascii=True),
                    "main_terms": json.dumps([item["term"] for item in (evidence or [])[:5]], ensure_ascii=True),
                    "competing_clusters": json.dumps(
                        [
                            {"cluster_id": cid, "confidence": float(score)}
                            for cid, score in cluster_similarity_rows[1:4]
                        ],
                        ensure_ascii=True,
                    ),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(explanations, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    pd.DataFrame(document_rows).to_parquet(args.doc_output, index=False)
    print(f"Wrote cluster explanations to {args.output}")
    print(f"Wrote document explanations to {args.doc_output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strapi_utils import data_dir, default_export_path, flatten_text_values, node_type_for_type, preferred_name, text_fields


def load_export(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Expected top-level export JSON object keyed by Strapi content type")
    return data


def document_payload(entity_type: str, record: dict[str, Any]) -> dict[str, Any] | None:
    rid = record.get("id")
    if rid is None:
        return None
    extracted = text_fields(record)
    chunks: list[str] = []
    name = preferred_name(record)
    if name:
        chunks.append(name)
    chunks.extend(extracted.values())
    text = flatten_text_values(chunks)
    if not text:
        return None
    return {
        "id": f"{entity_type}_{rid}",
        "type": entity_type,
        "text": text,
        "metadata": {
            "source_type": entity_type,
            "record_id": rid,
            "text_fields": sorted(extracted.keys()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NLP text corpus and embeddings from Strapi export")
    parser.add_argument("--input", type=Path, default=default_export_path())
    parser.add_argument("--documents", type=Path, default=data_dir() / "text_documents.json")
    parser.add_argument("--embeddings", type=Path, default=data_dir() / "embeddings.npy")
    parser.add_argument("--parquet", type=Path, default=data_dir() / "document_embeddings.parquet")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    export_data = load_export(args.input)
    documents: list[dict[str, Any]] = []
    for content_type, records in export_data.items():
        entity_type = node_type_for_type(content_type)
        for record in records:
            payload = document_payload(entity_type, record)
            if payload:
                documents.append(payload)

    args.documents.parent.mkdir(parents=True, exist_ok=True)
    with args.documents.open("w", encoding="utf-8") as handle:
        json.dump(documents, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    texts = [doc["text"] for doc in documents]
    if texts:
        model = SentenceTransformer(args.model)
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.asarray(vectors, dtype=np.float32)
    else:
        embeddings = np.empty((0, 384), dtype=np.float32)

    np.save(args.embeddings, embeddings)

    table = pa.table(
        {
            "id": [doc["id"] for doc in documents],
            "type": [doc["type"] for doc in documents],
            "text": texts,
            "metadata_json": [json.dumps(doc["metadata"], ensure_ascii=True, sort_keys=True) for doc in documents],
            "embedding": pa.array([list(map(float, row)) for row in embeddings], type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, args.parquet)

    print(f"Wrote text documents: {args.documents}")
    print(f"Wrote embedding matrix: {args.embeddings}")
    print(f"Wrote embedding parquet: {args.parquet}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strapi_utils import data_dir, default_export_path, flatten_text_values, node_type_for_type, preferred_name, text_fields


TEXT_KEYS = {
    "description",
    "text",
    "transcription",
    "interpretation",
    "perception",
    "strategic_plan",
    "plan",
    "strategy",
    "mission",
    "vision",
    "summary",
    "content",
    "body",
    "notes",
    "note",
    "memo",
    "analysis",
}


def load_export(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Expected top-level export JSON object keyed by Strapi content type")
    return data


def stringify_relation_links(value: Any, known_types: set[str]) -> tuple[list[str], list[str]]:
    agents: list[str] = []
    projects: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if item.get("id") is not None and (item.get("type") or item.get("__type") or item.get("entity_type")):
                entity_type = str(item.get("type") or item.get("__type") or item.get("entity_type")).lower()
                ref = f"{entity_type}_{item['id']}"
                if "agent" in entity_type:
                    agents.append(ref)
                if "project" in entity_type:
                    projects.append(ref)
            for key, nested in item.items():
                if key in {"data", "attributes", "relation", "relations", "items"}:
                    visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return sorted(set(agents)), sorted(set(projects))


def extract_text(record: dict[str, Any], content_type: str) -> str | None:
    pieces: list[str] = []
    name = preferred_name(record)
    if name:
        pieces.append(name)
    extracted = text_fields(record)
    pieces.extend(extracted.values())
    for key, value in record.items():
        if key.lower() in TEXT_KEYS and isinstance(value, str):
            pieces.append(value.strip())
    text = flatten_text_values(pieces)
    return text or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create traceable perception documents from Strapi export")
    parser.add_argument("--input", type=Path, default=default_export_path())
    parser.add_argument("--output", type=Path, default=data_dir() / "documents.parquet")
    parser.add_argument("--json-output", type=Path, default=data_dir() / "documents.json")
    args = parser.parse_args()

    export_data = load_export(args.input)
    known_types = {node_type_for_type(key) for key in export_data.keys()}
    documents: list[dict[str, Any]] = []

    for content_type, records in export_data.items():
        entity_type = node_type_for_type(content_type)
        for record in records:
            entity_id = record.get("id")
            if entity_id is None:
                continue
            text = extract_text(record, content_type)
            if not text:
                continue

            related_agents, related_projects = stringify_relation_links(record, known_types)
            metadata = {
                "source_content_type": content_type,
                "entity_type": entity_type,
                "record_id": entity_id,
                "text_fields": sorted(text_fields(record).keys()),
            }
            documents.append(
                {
                    "id": f"{entity_type}_{entity_id}",
                    "entity_type": entity_type,
                    "entity_id": int(entity_id) if isinstance(entity_id, int) else entity_id,
                    "text": text,
                    "related_agents": related_agents,
                    "related_projects": related_projects,
                    "metadata": metadata,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w", encoding="utf-8") as handle:
        json.dump(documents, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    table = pa.Table.from_pydict(
        {
            "id": [doc["id"] for doc in documents],
            "entity_type": [doc["entity_type"] for doc in documents],
            "entity_id": [doc["entity_id"] for doc in documents],
            "text": [doc["text"] for doc in documents],
            "related_agents": [json.dumps(doc["related_agents"], ensure_ascii=True) for doc in documents],
            "related_projects": [json.dumps(doc["related_projects"], ensure_ascii=True) for doc in documents],
            "metadata": [json.dumps(doc["metadata"], ensure_ascii=True, sort_keys=True) for doc in documents],
        }
    )
    pq.write_table(table, args.output)
    print(f"Wrote {len(documents)} documents to {args.output}")


if __name__ == "__main__":
    main()
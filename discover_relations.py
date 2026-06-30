from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from strapi_utils import choose_target_type, default_export_path, node_type_for_type, project_root, singularize


def load_export(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Expected top-level export JSON object keyed by Strapi content type")
    return data


def infer_target_type(field_name: str, value: Any, known_types: set[str]) -> str | None:
    singular = singularize(field_name)
    if singular in known_types:
        return singular

    if isinstance(value, dict):
        if value.get("count") is not None:
            return choose_target_type(field_name, known_types)
        data = value.get("data")
        if isinstance(data, dict) and data.get("id") is not None:
            return choose_target_type(field_name, known_types)
        if isinstance(data, list) and data:
            return choose_target_type(field_name, known_types)
        if value.get("id") is not None:
            return choose_target_type(field_name, known_types)

    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, (dict, int)):
            return choose_target_type(field_name, known_types)

    return None


def build_relation_schema(export_data: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, str]]:
    known_types = {node_type_for_type(key) for key in export_data.keys()}
    schema: dict[str, dict[str, str]] = {}
    for content_type, rows in export_data.items():
        source_type = node_type_for_type(content_type)
        fields: dict[str, str] = {}
        for row in rows:
            for field_name, value in row.items():
                target_type = infer_target_type(field_name, value, known_types)
                if target_type:
                    fields[field_name] = target_type
        if fields:
            schema[source_type] = dict(sorted(fields.items()))
    return schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer relation schema from Strapi export records")
    parser.add_argument("--input", type=Path, default=default_export_path())
    parser.add_argument("--output", type=Path, default=project_root() / "data" / "relation_schema.json")
    args = parser.parse_args()

    export_data = load_export(args.input)
    schema = build_relation_schema(export_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
    print(f"Wrote relation schema to {args.output}")


if __name__ == "__main__":
    main()

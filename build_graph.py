from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import networkx as nx

from discover_relations import build_relation_schema, load_export
from strapi_utils import (
    data_dir,
    default_export_path,
    flatten_text_values,
    json_dumps,
    node_type_for_type,
    preferred_name,
    safe_list,
)


def extract_targets(value: Any) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for item in safe_list(value):
        if isinstance(item, dict):
            if "data" in item:
                nested = item["data"]
                if isinstance(nested, list):
                    for n in nested:
                        if isinstance(n, dict) and n.get("id") is not None:
                            targets.append(n)
                elif isinstance(nested, dict) and nested.get("id") is not None:
                    targets.append(nested)
            elif item.get("id") is not None:
                targets.append(item)
        elif isinstance(item, int):
            targets.append({"id": item})
    return targets


def node_id(entity_type: str, entity_id: Any) -> str:
    return f"{entity_type}_{entity_id}"


def build_node_attributes(record: dict[str, Any], content_type: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "strapi_id": record.get("id"),
        "entity_type": node_type_for_type(content_type),
    }
    name = preferred_name(record)
    if name:
        attrs["name"] = name
    text_values = []
    for key in ("description", "text", "transcription", "interpretation", "perception"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            text_values.append(value.strip())
    if text_values:
        attrs["text"] = flatten_text_values(text_values)

    for key, value in record.items():
        if key == "id":
            continue
        if isinstance(value, (str, int, float, bool)):
            attrs[key] = value
        elif isinstance(value, dict) and set(value.keys()) == {"count"}:
            attrs[key] = value.get("count")
        else:
            attrs[f"{key}_json"] = json_dumps(value)
    return attrs


def build_edges_for_record(
    source_type: str,
    source_id: Any,
    record: dict[str, Any],
    relation_map: dict[str, str],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    source = node_id(source_type, source_id)
    for field_name, target_type in relation_map.items():
        value = record.get(field_name)
        if value is None:
            continue

        targets = extract_targets(value)
        for tgt in targets:
            tgt_id = tgt.get("id")
            if tgt_id is None:
                continue
            edges.append(
                {
                    "source": source,
                    "target": node_id(target_type, tgt_id),
                    "relation": field_name,
                    "source_type": source_type,
                    "target_type": target_type,
                }
            )

    return edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Build node/edge graph artifacts from Strapi export")
    parser.add_argument("--input", type=Path, default=default_export_path())
    parser.add_argument("--relation-schema", type=Path, default=data_dir() / "relation_schema.json")
    parser.add_argument("--nodes", type=Path, default=data_dir() / "nodes.csv")
    parser.add_argument("--edges", type=Path, default=data_dir() / "edges.csv")
    parser.add_argument("--graph", type=Path, default=data_dir() / "graph.graphml")
    args = parser.parse_args()

    export_data = load_export(args.input)
    if args.relation_schema.exists():
        relation_schema = json.loads(args.relation_schema.read_text(encoding="utf-8"))
    else:
        relation_schema = build_relation_schema(export_data)
        args.relation_schema.parent.mkdir(parents=True, exist_ok=True)
        args.relation_schema.write_text(json.dumps(relation_schema, indent=2, ensure_ascii=True), encoding="utf-8")

    graph = nx.MultiDiGraph()
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    for content_type, records in export_data.items():
        source_type = node_type_for_type(content_type)
        relation_map = relation_schema.get(source_type, {})
        for record in records:
            rid = record.get("id")
            if rid is None:
                continue
            nid = node_id(source_type, rid)
            attrs = build_node_attributes(record, content_type)
            graph.add_node(nid, **attrs)
            node_rows.append({"id": nid, **attrs})

            edges = build_edges_for_record(source_type, rid, record, relation_map)
            for edge in edges:
                edge_rows.append(edge)
                if edge["target"] not in graph:
                    graph.add_node(edge["target"], entity_type=edge["target_type"])
                graph.add_edge(
                    edge["source"],
                    edge["target"],
                    relation=edge["relation"],
                    source_type=edge["source_type"],
                    target_type=edge["target_type"],
                )

    args.nodes.parent.mkdir(parents=True, exist_ok=True)
    node_fields = sorted({k for row in node_rows for k in row.keys()}) if node_rows else ["id"]
    with args.nodes.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=node_fields)
        writer.writeheader()
        writer.writerows(node_rows)

    with args.edges.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "target", "relation", "source_type", "target_type"])
        writer.writeheader()
        writer.writerows(edge_rows)

    nx.write_graphml(graph, args.graph)
    print(f"Wrote nodes CSV: {args.nodes}")
    print(f"Wrote edges CSV: {args.edges}")
    print(f"Wrote graph GraphML: {args.graph}")


if __name__ == "__main__":
    main()

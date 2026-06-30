from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


TEXT_FIELD_HINTS = {
    "description",
    "text",
    "transcription",
    "interpretation",
    "perception",
    "summary",
    "content",
    "body",
    "notes",
    "note",
    "quote",
    "message",
    "title",
    "name",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_export_path() -> Path:
    return project_root() / "urdaibai-full-export.json"


def data_dir() -> Path:
    return project_root() / "data"


def normalize_content_type_key(key: str) -> str:
    tail = key.split(".")[-1]
    tail = tail.replace("-", "_")
    tail = re.sub(r"[^0-9a-zA-Z_]+", "_", tail)
    return tail.strip("_").lower()


def singularize(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name.lower()).strip("_")
    if name.endswith("ies") and len(name) > 3:
        return name[:-3] + "y"
    if name.endswith("sses"):
        return name[:-2]
    if name.endswith("s") and not name.endswith("ss") and len(name) > 3:
        return name[:-1]
    return name


def table_name_for_type(content_type: str) -> str:
    return normalize_content_type_key(content_type)


def node_type_for_type(content_type: str) -> str:
    return normalize_content_type_key(content_type)


def sanitize_identifier(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "field"
    if cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    return cleaned


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def coerce_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    return value


def preferred_name(record: dict[str, Any]) -> str | None:
    for key in ("name", "title", "label", "slug", "headline"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def text_fields(record: dict[str, Any]) -> dict[str, str]:
    extracted: dict[str, str] = {}
    for key, value in record.items():
        if not isinstance(value, str):
            continue
        key_l = key.lower()
        if key_l in TEXT_FIELD_HINTS or any(h in key_l for h in TEXT_FIELD_HINTS) or len(value.strip()) >= 120:
            stripped = value.strip()
            if stripped:
                extracted[key] = stripped
    return extracted


def choose_target_type(field_name: str, candidate_types: Iterable[str]) -> str | None:
    candidates = list(candidate_types)
    if not candidates:
        return None

    normalized_field = singularize(field_name)
    if normalized_field in candidates:
        return normalized_field

    field_tokens = set(normalized_field.split("_"))
    best_type = None
    best_score = -1
    for candidate in candidates:
        candidate_tokens = set(candidate.split("_"))
        score = len(field_tokens & candidate_tokens)
        if normalized_field == candidate:
            score += 5
        elif normalized_field.startswith(candidate) or candidate.startswith(normalized_field):
            score += 2
        if score > best_score:
            best_type = candidate
            best_score = score
    return best_type


def flatten_text_values(values: Iterable[str]) -> str:
    parts = [p.strip() for p in values if p and p.strip()]
    return "\n\n".join(parts)


def safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]

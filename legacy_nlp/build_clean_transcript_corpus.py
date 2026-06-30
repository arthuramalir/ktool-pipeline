from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strapi_utils import data_dir


INNER_FIELDS = ["id", "code", "date", "locale", "createdAt", "updatedAt", "transcript"]
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[?.!])(?=\s*[A-ZÁÉÍÓÚÑÜ¿¡(])")


def load_wrapped_export(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";", engine="python")
    if frame.empty:
        raise ValueError(f"No rows found in {path}")
    return frame


def parse_inner_row(raw_row: str) -> dict[str, Any]:
    parsed = next(csv.reader(io.StringIO(str(raw_row)), delimiter=",", quotechar='"', doublequote=True))
    if len(parsed) != len(INNER_FIELDS):
        raise ValueError(f"Expected {len(INNER_FIELDS)} fields in wrapped transcript row, got {len(parsed)}")
    return dict(zip(INNER_FIELDS, parsed))


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    parts = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    if len(parts) <= 1:
        return parts
    merged: list[str] = []
    for part in parts:
        if merged and len(part) < 20 and not part.endswith("?") and part[0:1].islower():
            merged[-1] = f"{merged[-1]} {part}".strip()
        else:
            merged.append(part)
    return merged


def looks_like_prompt(sentence: str) -> bool:
    text = sentence.strip()
    if not text.endswith("?"):
        return False

    lowered = text.lower()
    prompt_markers = (
        "zer ",
        "zein ",
        "zelan ",
        "nola ",
        "zergatik ",
        "noiz ",
        "non ",
        "nor ",
        "qué ",
        "como ",
        "cómo ",
        "por qué ",
        "cuál ",
        "cuáles ",
        "dónde ",
        "cuándo ",
        "who ",
        "what ",
        "why ",
        "how ",
        "which ",
        "where ",
        "when ",
        "can you ",
        "could you ",
        "would you ",
    )
    if any(lowered.startswith(marker) for marker in prompt_markers):
        return True

    if lowered.startswith(("aipatzen ", "testuinguru honetan", "lehen esaten", "en este contexto", "en este sentido", "in this context")):
        return True

    return len(text) <= 400


def clean_transcript(text: str) -> tuple[str, list[dict[str, str]]]:
    segments = split_sentences(text)
    turns: list[dict[str, str]] = []
    current_prompt: str | None = None
    answer_parts: list[str] = []

    for segment in segments:
        if looks_like_prompt(segment):
            if current_prompt is not None and answer_parts:
                answer_text = normalize_whitespace(" ".join(answer_parts))
                if answer_text:
                    turns.append({"prompt": current_prompt, "answer": answer_text})
            current_prompt = segment.strip()
            answer_parts = []
        else:
            answer_parts.append(segment.strip())

    if current_prompt is not None and answer_parts:
        answer_text = normalize_whitespace(" ".join(answer_parts))
        if answer_text:
            turns.append({"prompt": current_prompt, "answer": answer_text})

    cleaned_text = normalize_whitespace("\n\n".join(turn["answer"] for turn in turns))
    return cleaned_text, turns


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an answer-only transcript corpus from the wrapped export")
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "FINAL_transcriptions_export.csv")
    parser.add_argument("--output-csv", type=Path, default=data_dir() / "cleaned_transcripts.csv")
    parser.add_argument("--output-documents", type=Path, default=data_dir() / "cleaned_transcript_documents.parquet")
    args = parser.parse_args()

    wrapped = load_wrapped_export(args.input)
    cleaned_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []

    for index, raw_row in wrapped.iloc[:, 0].items():
        parsed = parse_inner_row(raw_row)
        cleaned_text, turns = clean_transcript(parsed["transcript"])
        turn_count = len(turns)
        prompt_count = sum(1 for turn in turns if turn.get("prompt"))

        cleaned_row = {
            "id": parsed["id"],
            "code": parsed["code"],
            "date": parsed["date"],
            "locale": parsed["locale"],
            "createdAt": parsed["createdAt"],
            "updatedAt": parsed["updatedAt"],
            "cleaned_transcript": cleaned_text,
            "prompt_count": prompt_count,
            "answer_turn_count": turn_count,
            "turns_json": json.dumps(turns, ensure_ascii=True),
        }
        cleaned_rows.append(cleaned_row)

        document_rows.append(
            {
                "id": f"transcription_clean_{parsed['id']}",
                "entity_type": "transcription_clean",
                "entity_id": int(parsed["id"]) if str(parsed["id"]).isdigit() else parsed["id"],
                "text": cleaned_text,
                "related_agents": json.dumps([], ensure_ascii=True),
                "related_projects": json.dumps([], ensure_ascii=True),
                "metadata": json.dumps(
                    {
                        "source_code": parsed["code"],
                        "source_locale": parsed["locale"],
                        "source_date": parsed["date"],
                        "prompt_count": prompt_count,
                        "answer_turn_count": turn_count,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            }
        )

    cleaned_frame = pd.DataFrame(cleaned_rows)
    document_frame = pd.DataFrame(document_rows)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned_frame.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    table = pa.Table.from_pydict({column: document_frame[column].tolist() for column in document_frame.columns})
    pq.write_table(table, args.output_documents)

    print(f"Wrote cleaned transcripts to {args.output_csv}")
    print(f"Wrote cleaned quote documents to {args.output_documents}")


if __name__ == "__main__":
    main()
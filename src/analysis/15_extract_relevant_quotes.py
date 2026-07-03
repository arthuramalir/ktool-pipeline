from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173_synthetic")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
BASE_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR
ANALYSIS_DIR = BASE_DIR / "analysis"

ACTION_MARKERS = re.compile(r"\b(helped|improved|supports?|creates?|builds?|enables?|reduces?|causes?|because|therefore|opportunity|challenge|impact|prototype|pilot|project)\b", re.IGNORECASE)
VALUE_MARKERS = re.compile(r"\b(value|community|equity|inclusion|wellbeing|learning|innovation|sustainability|health)\b", re.IGNORECASE)
CONTRADICTION_MARKERS = re.compile(r"\b(but|however|although|yet|instead|despite|problem|barrier|risk|gap|conflict)\b", re.IGNORECASE)


def load_informations() -> pd.DataFrame:
    candidates = [BASE_DIR / "informations.csv", BASE_DIR / "entities" / "informations.csv"]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    return pd.DataFrame()


def safe_text(*values: object) -> str:
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none"}:
            return text
    return ""


def score_quote(text: str, row: pd.Series) -> tuple[float, str]:
    score = 0.0
    reasons = []
    length = len(text)
    if length >= 40:
        score += 0.2
        reasons.append("substantive length")
    if length >= 100:
        score += 0.15
        reasons.append("detail rich")
    if ACTION_MARKERS.search(text):
        score += 0.25
        reasons.append("action / cause language")
    if VALUE_MARKERS.search(text):
        score += 0.2
        reasons.append("value language")
    if CONTRADICTION_MARKERS.search(text):
        score += 0.2
        reasons.append("contrast / tension language")
    if safe_text(row.get("channel_id")):
        score += 0.1
        reasons.append("channel anchored")
    if safe_text(row.get("perception_ids")):
        score += 0.1
        reasons.append("perception linked")
    if safe_text(row.get("topics_sub_areas")) or safe_text(row.get("topics_thematic_areas")):
        score += 0.1
        reasons.append("theme linked")
    return min(score, 1.0), "; ".join(reasons) if reasons else "baseline quote"


def extract_quotes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        text = safe_text(row.get("quote"), row.get("description"), row.get("title"), row.get("name"))
        if not text:
            continue
        score, reasons = score_quote(text, row)
        rows.append(
            {
                "information_id": safe_text(row.get("global_id"), row.get("native_id")),
                "channel_id": safe_text(row.get("channel_id")),
                "channel_code": safe_text(row.get("channel_code"), row.get("channel_name")),
                "quote": text,
                "description": safe_text(row.get("description")),
                "topics_sub_areas": safe_text(row.get("topics_sub_areas")),
                "topics_thematic_areas": safe_text(row.get("topics_thematic_areas")),
                "values": safe_text(row.get("values")),
                "selection_score": round(score, 4),
                "selection_reason": reasons,
                "source_node_type": safe_text(row.get("node_type")),
                "node_label": safe_text(row.get("label"), row.get("title"), row.get("name")),
                "perception_ids": safe_text(row.get("perception_ids")),
                "challenge_ids": safe_text(row.get("challenge_id"), row.get("challenge_ids")),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["selection_score", "information_id"], ascending=[False, True]).reset_index(drop=True)
        out["quote_id"] = [f"quote_{i + 1}" for i in range(len(out))]
    return out


def main() -> None:
    df = load_informations()
    if df.empty:
        print("[ERROR] No informations table found for quote extraction.")
        return

    quotes = extract_quotes(df)
    if quotes.empty:
        print("[WARNING] No quote candidates extracted.")
        return

    (BASE_DIR / "relationships").mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    quotes.to_csv(ANALYSIS_DIR / "quote_candidates.csv", index=False)
    quotes.to_csv(BASE_DIR / "relationships" / "quote_candidates.csv", index=False)
    print(f"Extracted {len(quotes)} quote candidates.")


if __name__ == "__main__":
    main()

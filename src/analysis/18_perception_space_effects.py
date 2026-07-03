from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173_synthetic")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
BASE_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR
ANALYSIS_DIR = BASE_DIR / "analysis"


def safe_text(*values: object) -> str:
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text.lower() not in {"", "nan", "none"}:
            return text
    return ""


def load_csv(paths: list[Path]) -> pd.DataFrame:
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    return pd.DataFrame()


def split_items(value: object) -> set[str]:
    text = safe_text(value)
    if not text:
        return set()
    return {part.strip() for part in text.replace("|", ",").split(",") if part.strip()}


def load_nodes() -> pd.DataFrame:
    return load_csv([BASE_DIR / "nodes.csv", BASE_DIR / "analytics" / "nodes.csv"])


def load_recommendations() -> pd.DataFrame:
    return load_csv([
        ANALYSIS_DIR / "gnn" / "gnn_link_recommendations.csv",
        ANALYSIS_DIR / "gnn_link_recommendations.csv",
    ])


def node_lookup(nodes: pd.DataFrame) -> dict[str, pd.Series]:
    if nodes.empty or "global_id" not in nodes.columns:
        return {}
    return {safe_text(row["global_id"]): row for _, row in nodes.iterrows()}


def main() -> None:
    nodes = load_nodes()
    recs = load_recommendations()
    if nodes.empty or recs.empty:
        print("[ERROR] Missing nodes or GNN recommendations for perception-space effects.")
        return

    lookup = node_lookup(nodes)
    rows = []
    for _, rec in recs.iterrows():
        source_id = safe_text(rec.get("source_global_id"), rec.get("source"))
        target_id = safe_text(rec.get("target_global_id"), rec.get("target"))
        source = lookup.get(source_id)
        target = lookup.get(target_id)
        if source is None or target is None:
            continue

        source_themes = split_items(source.get("topics_thematic_areas")) | split_items(source.get("topics_sub_areas"))
        target_themes = split_items(target.get("topics_thematic_areas")) | split_items(target.get("topics_sub_areas"))
        source_perceptions = split_items(source.get("perception_ids"))
        target_perceptions = split_items(target.get("perception_ids"))
        source_values = split_items(source.get("values"))
        target_values = split_items(target.get("values"))

        shared_perceptions = source_perceptions.intersection(target_perceptions)
        shared_themes = source_themes.intersection(target_themes)
        shared_values = source_values.intersection(target_values)
        union_size = len(source_perceptions.union(target_perceptions))
        overlap_ratio = float(len(shared_perceptions) / union_size) if union_size else 0.0

        if shared_perceptions:
            effect_type = "Reinforces shared perception space"
            effect_note = "Both endpoints already sit near the same perception set."
        elif shared_themes or shared_values:
            effect_type = "Bridges narrative framing"
            effect_note = "The proposed mapping link may bridge thematically aligned but not perception-aligned nodes."
        else:
            effect_type = "Likely structural only"
            effect_note = "The proposed link is structurally useful but currently has little evidence of perception overlap."

        rows.append(
            {
                "source_global_id": source_id,
                "target_global_id": target_id,
                "link_type": safe_text(rec.get("link_type"), rec.get("pair_kind")),
                "pair_kind": safe_text(rec.get("pair_kind")),
                "mapping_score": float(rec.get("score", rec.get("prediction_score", 0)) or 0),
                "source_themes": " | ".join(sorted(source_themes)),
                "target_themes": " | ".join(sorted(target_themes)),
                "shared_themes": " | ".join(sorted(shared_themes)),
                "source_perceptions": " | ".join(sorted(source_perceptions)),
                "target_perceptions": " | ".join(sorted(target_perceptions)),
                "shared_perceptions": " | ".join(sorted(shared_perceptions)),
                "source_values": " | ".join(sorted(source_values)),
                "target_values": " | ".join(sorted(target_values)),
                "shared_values": " | ".join(sorted(shared_values)),
                "perception_overlap_ratio": round(overlap_ratio, 4),
                "perception_bridge_score": round(float(len(shared_themes) + len(shared_values) + len(shared_perceptions) * 1.5), 4),
                "effect_type": effect_type,
                "effect_note": effect_note,
            }
        )

    out = pd.DataFrame(rows)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(ANALYSIS_DIR / "perception_effects.csv", index=False)
    print(f"Wrote {len(out)} perception-space effect rows.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from graph_utils import ANALYSIS_DIR, DATA_DIR, ensure_output_dirs, write_json


CRITICAL_RELATION_FILES = [
    "initiative_perception_links.csv",
    "initiative_thematic_area_links.csv",
    "initiative_lead_agent_links.csv",
    "agent_initiative_links.csv",
    "channel_information_links.csv",
    "information_pattern_links.csv",
    "information_value_links.csv",
]

MIN_ROW_THRESHOLD = {
    "information_value_links.csv": 1,
    "channel_information_links.csv": 1,
    "information_pattern_links.csv": 1,
    "initiative_perception_links.csv": 1,
    "initiative_thematic_area_links.csv": 1,
    "initiative_lead_agent_links.csv": 1,
    "agent_initiative_links.csv": 1,
}


def count_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        frame = pd.read_csv(path)
        return int(len(frame.index))
    except Exception:
        return 0


def resolve_file(name: str) -> Path:
    root_path = DATA_DIR / name
    if root_path.exists():
        return root_path

    rel_path = DATA_DIR / "relationships" / name
    if rel_path.exists():
        return rel_path

    return root_path


def run_quality_gate() -> None:
    ensure_output_dirs()

    checks = []
    failing = []

    for file_name in CRITICAL_RELATION_FILES:
        path = resolve_file(file_name)
        rows = count_rows(path)
        expected_min = MIN_ROW_THRESHOLD.get(file_name, 1)
        passed = rows >= expected_min
        check = {
            "file": file_name,
            "path": str(path),
            "rows": rows,
            "expected_min_rows": expected_min,
            "passed": passed,
        }
        checks.append(check)
        if not passed:
            failing.append(check)

    report = {
        "quality_gate": "critical_relations",
        "dataset_dir": str(DATA_DIR),
        "total_checks": len(checks),
        "failed_checks": len(failing),
        "status": "pass" if not failing else "fail",
        "checks": checks,
    }

    write_json(ANALYSIS_DIR / "relation_quality_gate.json", report)

    print("Quality gate generated: relation_quality_gate.json")
    print(f"Checks: {len(checks)} | Failing: {len(failing)}")
    if failing:
        print("Failing relation files:")
        for item in failing:
            print(
                " - "
                f"{item['file']} rows={item['rows']} expected>={item['expected_min_rows']}"
            )


if __name__ == "__main__":
    run_quality_gate()

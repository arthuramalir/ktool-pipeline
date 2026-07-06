"""Run core analytical scripts for active platforms, skipping disabled ones.

Usage examples (PowerShell):
    $env:KTOOL_PLATFORM_ID="173"
    $env:KTOOL_OUTPUT_SUBDIR="test"
    python src/analysis/01_run_all_platform.py

Optional:
    $env:KTOOL_DISABLED_ANALYTICS_PLATFORMS="155"
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173_synthetic")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
DISABLED_PLATFORMS = {
    p.strip() for p in os.environ.get("KTOOL_DISABLED_ANALYTICS_PLATFORMS", "155").split(",") if p.strip()
}

ANALYSIS_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

CORE_SCRIPTS = [
    ("00_quality_gate_relations.py", "Quality gate for relation coverage"),
    ("01_graph_readiness_audit.py", "Graph audit - counts, components, density"),
    ("02_graph_structural_possibility.py", "Centrality, articulation points, bridges, k-core"),
    ("05_robustness.py", "Node-removal decay simulation"),
    ("06_value_proof_comparison.py", "Record-vs-graph contrast matrix"),
    ("09_perception_diagnostics.py", "Perception diagnostics — coherence, purity, source entropy"),
    ("21_extract_narrative_layers.py", "Narrative extraction — surface/implicit claims + metanarrative classification"),
    ("00_prepare_graph_tables.py", "Re-prepare graph with claim nodes/edges from narrative extraction"),
    ("11_gnn_preparation.py", "GNN preparation - features, heterographs, uncertainty-aware confidence"),
    ("12_train_gnn_node_type.py", "GNN node-type benchmark"),
    ("13_train_gnn_link_prediction.py", "GNN link prediction and mapping AI shortlist"),
    ("14_structural_impact_prediction.py", "Structural impact prediction — issue-driven edge recommendations + narrative impact"),
]

PLATFORM_SPECIFIC = {
    "10": [
        ("04_investment_opportunity_p10.py", "Investment analysis for platform 10"),
    ],
    "173": [
        ("04_investment_opportunity.py", "Investment analysis for platform 173"),
    ],
    "173_synthetic": [
        ("00_enrich_173_synthetic_dataset.py", "Synthetic enrichment for sparse initiative, listening, and edge fields"),
        ("04_investment_oppurtunity.py", "Investment analysis for synthetic platform 173"),
        ("04_investment_opportunity_synthetic.py", "Financial simulation — value leverage, stranded assets, financial diffusion"),
        ("05_prototype_project_candidates_synthetic.py", "Prototype-project candidate scoring"),
        ("15_extract_relevant_quotes.py", "Listening-layer quote extraction"),
        ("16_detect_quote_semantic_edges.py", "Quote semantic-edge detection"),
        ("17_cluster_quotes_into_profiles.py", "Quote clustering and narrative profiles"),
        ("18_perception_space_effects.py", "Perception-space effects for mapping proposals"),
    ],
}


def run_script(script_name: str, description: str) -> int:
    path = ROOT / "src" / "analysis" / script_name
    if not path.exists():
        print(f"SKIP {script_name}: not found")
        return 0

    print("\n" + "=" * 60)
    print(f"RUNNING: {script_name} - {description}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")

    if result.returncode != 0:
        print(f"  ERROR (rc={result.returncode})")
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines()[:12]:
                print(f"  {line}")
    return result.returncode


def main() -> None:
    if PLATFORM_ID in DISABLED_PLATFORMS:
        print("=" * 60)
        print(f"Analytics runner skipped for platform {PLATFORM_ID} (disabled by policy).")
        print(f"Disabled platforms: {sorted(DISABLED_PLATFORMS)}")
        print("=" * 60)
        return

    if PLATFORM_ID == "173_synthetic":
        scripts = [
            ("00_enrich_173_synthetic_dataset.py", "Synthetic enrichment for sparse initiative, listening, and edge fields"),
            ("15_extract_relevant_quotes.py", "Listening-layer quote extraction"),
            ("16_detect_quote_semantic_edges.py", "Quote semantic-edge detection"),
            ("17_cluster_quotes_into_profiles.py", "Quote clustering and narrative profiles"),
            ("00_prepare_graph_tables.py", "Rebuild the normalized graph tables from the enriched synthetic dataset"),
        ]
        scripts.extend(CORE_SCRIPTS)
        scripts.extend(
            [
                ("04_investment_opportunity.py", "Investment analysis for synthetic platform 173"),
                ("04_investment_opportunity_synthetic.py", "Financial simulation — value leverage, stranded assets, financial diffusion"),
                ("05_prototype_project_candidates_synthetic.py", "Prototype-project candidate scoring"),
                ("18_perception_space_effects.py", "Perception-space effects for mapping proposals"),
            ]
        )
    else:
        scripts = list(CORE_SCRIPTS)
        scripts.extend(PLATFORM_SPECIFIC.get(PLATFORM_ID, []))

    failures = 0
    for script_name, description in scripts:
        failures += int(run_script(script_name, description) != 0)

    print("\n" + "=" * 60)
    print("ANALYTICS RUN COMPLETE")
    print(f"Platform: {PLATFORM_ID} | Output: {OUTPUT_SUBDIR}")
    print(f"Outputs in: {ANALYSIS_DIR}")
    print(f"Failed scripts: {failures}")
    print("=" * 60)


if __name__ == "__main__":
    main()

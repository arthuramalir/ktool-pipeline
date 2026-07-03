"""Run all analytical scripts for Platform 10 (Chile COPOLAD).

Usage:
    set KTOOL_PLATFORM_ID=10 & set KTOOL_OUTPUT_SUBDIR=test
    python src/analysis/01_run_all_p10.py

Requires: 00_prepare_graph_tables_p10.py already run.
"""

import os, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "10")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
ANALYSIS_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    # Graph audit
    ("01_graph_readiness_audit.py", "Graph audit — counts, components, density"),
    # Structural analysis
    ("02_graph_structural_possibility.py", "Centrality, articulation points, bridges, k-core"),
    # Investment analysis (platform-10 specific)
    ("04_investment_opportunity_p10.py", "Bridge scores, challenge weights, investment analysis with labelled investment data"),
    # Robustness (may be slow on 1150 nodes)
    ("05_robustness.py", "Node-removal decay simulation"),
    # Value proof
    ("06_value_proof_comparison.py", "Record-vs-graph contrast matrix"),
]

for script, desc in SCRIPTS:
    path = ROOT / "src" / "analysis" / script
    if not path.exists():
        print(f"SKIP {script}: not found")
        continue
    print(f"\n{'='*60}")
    print(f"RUNNING: {script} — {desc}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")
    else:
        print(f"  ERROR (rc={result.returncode})")
        for line in result.stderr.strip().split('\n')[:10]:
            print(f"  {line}")

print(f"\n{'='*60}")
print("ALL DONE")
print(f"Outputs in: {ANALYSIS_DIR}")

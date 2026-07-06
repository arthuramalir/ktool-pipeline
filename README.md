# KTool Pipeline

Extract, build, and analyze ecosystem graphs from KTool stakeholder engagement data.

## Structure

| Directory | Purpose |
|-----------|---------|
| `src/app.py` | Streamlit dashboard for government decision-makers |
| `src/analysis/` | Pipeline: graph extraction, semantic edges, narrative claims, GNN experiments |
| `src/gephi_filters/` | Filtered GEXF/GraphML exports for Gephi |
| `src/MAIN_comprehensive_pipeline.py` | Run data extraction end-to-end |
| `src/MAIN_generate_ecosystem_network.py` | Build and export ecosystem GEXF |

## Usage

```powershell
# Full pipeline (extraction → semantic edges → graph → narrative claims → GNN)
$env:KTOOL_PLATFORM_ID = "173"
$env:KTOOL_OUTPUT_SUBDIR = "test"
python src/analysis/01_run_all_platform.py

# Or step by step:
$env:KTOOL_PLATFORM_ID = "173"
$env:KTOOL_OUTPUT_SUBDIR = "test"
python src/analysis/00_prepare_graph_tables.py
python src/analysis/21_extract_narrative_layers.py
python src/analysis/00_prepare_graph_tables.py
python src/analysis/11_gnn_preparation.py

# Dashboard
streamlit run src/app.py
```

## Environment Variables

- `KTOOL_PLATFORM_ID` — dataset ID (e.g. `173`, `173_synthetic`)
- `KTOOL_OUTPUT_SUBDIR` — sub-directory (typically `test` or `prod`)
- `KTOOL_PROJECT_NAME` — project name (default `ALC`)
- `KTOOL_DISABLED_ANALYTICS_PLATFORMS` — comma-separated platform IDs to skip

## Pipeline Stages

| Stage | Script | Artifacts |
|-------|--------|-----------|
| Data extraction | `MAIN_comprehensive_pipeline.py` | Entity/relationship CSVs |
| Semantic edges | `14_alc_advanced_semantic_edges.py` | AI-inferred quote–quote edges |
| Narrative extraction | `21_extract_narrative_layers.py` | Surface/implicit claims, metanarratives |
| Graph normalization | `00_prepare_graph_tables.py` | `nodes.csv`, `edges.csv` (incl. claim nodes) |
| GNN preparation | `11_gnn_preparation.py` | Node features, edge tensors |
| GNN training | `12_` / `13_` | Node-type / link-prediction models |
| Dashboard | `src/app.py` | Streamlit UI (11 tabs) |

## Provenance

All AI-generated data (semantic edges, narrative claims) is explicitly tagged:

- **Nodes**: `is_ai_generated=True`, `generated_by=<script_name>`
- **Edges**: `is_ai_generated=True`, `edge_origin="ai_inferred"`, `generated_by=<script_name>`
- **Original data**: fields are empty/NaN — trivially distinguishable.

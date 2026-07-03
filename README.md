# KTool Pipeline

Extract, build, and analyze ecosystem graphs from KTool stakeholder engagement data.

## Structure

| Directory | Purpose |
|-----------|---------|
| `src/app.py` | Streamlit dashboard for government decision-makers |
| `src/analysis/` | Pipeline: graph extraction (00–07), metrics, semantic analysis (08–19) |
| `src/gephi_filters/` | Filtered GEXF/GraphML exports for Gephi |
| `src/MAIN_comprehensive_pipeline.py` | Run full pipeline end-to-end |
| `src/MAIN_generate_ecosystem_network.py` | Build and export ecosystem GEXF |

## Usage

```bash
pip install -r requirements.txt

# Build graph tables + export
python src/MAIN_comprehensive_pipeline.py

# Or step by step:
KTOOL_PLATFORM_ID=173 KTOOL_OUTPUT_SUBDIR=test python src/analysis/00_prepare_graph_tables.py
KTOOL_PLATFORM_ID=173 KTOOL_OUTPUT_SUBDIR=test python src/MAIN_generate_ecosystem_network.py
KTOOL_PLATFORM_ID=173 KTOOL_OUTPUT_SUBDIR=test python src/gephi_filters/generate_all_filters.py

# Dashboard
streamlit run src/app.py
```

## Environment Variables

- `KTOOL_PLATFORM_ID` — dataset ID (e.g. `173`, `173_synthetic`)
- `KTOOL_OUTPUT_SUBDIR` — sub-directory (typically `test` or `prod`)
- `KTOOL_PROJECT_NAME` — project name (default `ALC`)

# Gephi Filter Exports

Generates filtered GEXF + GraphML files for Gephi visualisation.

## Run

```powershell
python src/gephi_filters/generate_all_filters.py
```

## Output

All files go to `data/processed/173/test/analysis/gephi_filters/`.

Counts depend on the current graph and may differ from the table below (example values for 173/test with claims included):

| File | Content | Nodes | Edges |
|------|---------|-------|-------|
| `full_graph` | All edges | 497 | ~650 |
| `source_data_only` | No semantic or claim edges | 375 | ~150 |
| `semantic_only` | Only quote-quote edges | ~72 | ~190 |
| `claim_only` | Only narrative claim edges | ~200 | ~300 |
| `largest_component` | Largest connected component | depends | depends |
| `node_type_agent` | Subgraph around agents | ~50 | ~200 |
| `node_type_information` | Subgraph around information nodes | ~72 | ~150 |
| `node_type_claim` | Subgraph around claim nodes | ~122 | ~300 |

Open any `.graphml` or `.gexf` in Gephi. `filter_inventory.csv` has the full list.

## Edge type filtering

Use Gephi's edge filter to toggle:

- `is_ai_generated = False` — source-data only
- `edge_family = narrative_claim` — claim edges only
- `generated_by = 14_alc_advanced_semantic_edges.py` — semantic edges only

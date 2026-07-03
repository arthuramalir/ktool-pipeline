# Gephi Filter Exports

Generates filtered GEXF + GraphML files for Gephi visualisation.

## Run

```powershell
python src/gephi_filters/generate_all_filters.py
```

## Output

All files go to `data/processed/173/test/analysis/gephi_filters/`:

| File | Content | Nodes | Edges |
|------|---------|-------|-------|
| `full_graph` | All edges | 384 | 496 |
| `source_data_only` | No semantic edges | 384 | 433 |
| `semantic_only` | Only quote-quote edges | ~72 | 120 |
| `edge_type_similarity` | Similarity edges only | ~44 | 44 |
| `edge_type_contradiction` | Contradiction edges only | ~20 | 10 |
| `edge_type_causality` | Causality edges only | ~26 | 13 |
| `edge_type_sequence` | Sequence edges only | ~106 | 53 |
| `largest_component` | Largest connected component (33% of graph) | 126 | ~180 |
| `largest_component_source_only` | Largest component, no semantic | 126 | ~160 |
| `largest_component_semantic_only` | Largest component, semantic only | ~60 | ~50 |
| `node_type_agent` | Subgraph around agents | 136 | ~200 |
| `node_type_information` | Subgraph around information nodes | 72 | ~150 |
| `node_type_challenge` | Subgraph around challenges | 15 | ~30 |
| `agents_via_semantic` | Agents connected by semantic edges | ~30 | ~120 |
| `information_network` | Quote nodes and all connections | 72 | ~150 |

Open any `.graphml` or `.gexf` in Gephi. `filter_inventory.csv` has the full list.

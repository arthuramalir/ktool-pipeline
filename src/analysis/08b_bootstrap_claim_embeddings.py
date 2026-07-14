"""Bootstrap claim-node embeddings into the semantic embedding store.

Claim nodes (produced by 21_extract_narrative_layers.py) are created AFTER the
original 08_nlp_semantic_alignment.py ran, so they have zero embeddings. This
script loads their description text, runs sentence-transformer, and appends
the resulting vectors to node_semantic_embeddings.parquet.

Also adds text for ANY node missing from the embedding store to handle
future additions.

Usage:
    set KTOOL_PLATFORM_ID=173 & set KTOOL_OUTPUT_SUBDIR=test
    python src/analysis/08b_bootstrap_claim_embeddings.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from graph_utils import ANALYSIS_DIR, ANALYTICS_DIR, load_nodes_edges


def main() -> None:
    emb_path = ANALYSIS_DIR / "node_semantic_embeddings.parquet"
    claim_nodes_path = ANALYSIS_DIR / "narrative_layers" / "claim_nodes.csv"

    # ── Load existing embeddings ──
    if emb_path.exists():
        emb_df = pd.read_parquet(emb_path)
        existing_gids = set(emb_df["global_id"].values)
        print(f"Existing embeddings: {len(emb_df)} nodes")
    else:
        emb_df = pd.DataFrame(columns=["global_id", "embedding"])
        existing_gids = set()
        print("No existing embeddings — starting fresh")

    # ── Load all nodes (base + claims) ──
    nodes, _ = load_nodes_edges()
    all_gids = set(nodes["global_id"].values)
    missing_gids = all_gids - existing_gids
    print(f"All nodes: {len(all_gids)} | Missing from embeddings: {len(missing_gids)}")

    if not missing_gids:
        print("All nodes already have embeddings — nothing to do.")
        return

    # ── Gather text for missing nodes ──
    missing_nodes = nodes[nodes["global_id"].isin(missing_gids)].copy()
    text_map: dict[str, str] = {}

    # Try description first, then label
    for _, row in missing_nodes.iterrows():
        text = str(row.get("description", "") or "").strip()
        if not text:
            text = str(row.get("label", "") or "").strip()
        if text and text != "nan":
            text_map[row["global_id"]] = text

    # If claim_nodes.csv exists, use those descriptions as fallback
    if claim_nodes_path.exists():
        claim_df = pd.read_csv(claim_nodes_path)
        for _, row in claim_df.iterrows():
            gid = str(row["global_id"])
            if gid in missing_gids and gid not in text_map:
                text = str(row.get("description", "") or "").strip()
                if not text:
                    text = str(row.get("label", "") or "").strip()
                if text and text != "nan":
                    text_map[gid] = text

    if not text_map:
        print("No text available for any missing nodes — nothing to embed.")
        return

    print(f"Nodes with text to embed: {len(text_map)}")

    # ── Encode ──
    model = SentenceTransformer("all-MiniLM-L6-v2")
    gids_to_encode = list(text_map.keys())
    texts_to_encode = [text_map[gid] for gid in gids_to_encode]
    print(f"Encoding {len(texts_to_encode)} texts (batch size=32)...")
    new_embeddings = model.encode(texts_to_encode, show_progress_bar=True, batch_size=32)
    print(f"  → {new_embeddings.shape[0]} vectors of dim {new_embeddings.shape[1]}")

    # ── Build new rows ──
    new_rows = []
    for i, gid in enumerate(gids_to_encode):
        new_rows.append({"global_id": gid, "embedding": new_embeddings[i].tolist()})

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([emb_df, new_df], ignore_index=True)
    combined.to_parquet(emb_path, index=False)
    print(f"Updated embeddings: {len(combined)} nodes → {emb_path}")
    print(f"  Added: {len(new_df)} | Zero vectors remaining: {sum(np.linalg.norm(np.stack(combined['embedding'].values), axis=1) == 0)}")


if __name__ == "__main__":
    main()

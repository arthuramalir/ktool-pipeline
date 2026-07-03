"""Perception Diagnostics — Platform 173.

Computes quantitative health metrics for each perception archetype in the
K-Tool Listening module.  Uses the LLM-generated semantic edges
(quote_semantic_edges.csv) that are already in the relationships/ folder and
the informations + perceptions entity tables.

Metrics per perception:
  internal_coherence    — mean cosine similarity of LLM-edge weights between
                          quotes that share the same perception label
  purity_score          — fraction of each quote's semantic neighbours that
                          belong to the same perception (top-5 neighbours)
  source_entropy        — Shannon entropy over interview channels that fed this
                          perception (low = single channel dominates)
  contradiction_density — fraction of intra-perception semantic edges that are
                          of type contradiction
  quote_count           — raw number of quotes assigned to this perception
  status_flag           — human-readable health label

Usage:
  set KTOOL_PLATFORM_ID=173 & set KTOOL_OUTPUT_SUBDIR=test
  python src/analysis/09_perception_diagnostics.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_utils import ANALYSIS_DIR, DATA_DIR, write_frame, write_json

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RELS_DIR = DATA_DIR / "relationships"
ENTITIES_DIR = DATA_DIR / "entities"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def _load_df(candidates: list[Path]) -> pd.DataFrame:
    for p in candidates:
        if p.exists() and p.stat().st_size > 2:
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    info = _load_df([
        ENTITIES_DIR / "informations.csv",
        DATA_DIR / "informations.csv",
    ])

    perceptions = _load_df([
        ENTITIES_DIR / "perceptions.csv",
        DATA_DIR / "perceptions.csv",
    ])

    # LLM-generated semantic edges — try both locations
    sem_edges = _load_df([
        DATA_DIR / "relationships" / "quote_semantic_edges.csv",
        ANALYSIS_DIR / "quote_semantic_edges.csv",
    ])

    return info, perceptions, sem_edges


# ---------------------------------------------------------------------------
# Build perception -> quote membership
# ---------------------------------------------------------------------------

def build_perception_membership(info: pd.DataFrame, perceptions: pd.DataFrame) -> dict[str, list[str]]:
    """Map each perception id to the list of information global_ids.

    Strategy (in order of preference):
    1. Direct 'perception_id' / 'perception_global_id' column on informations.
    2. Keyword match between quotes' topics_thematic_areas and perception names.
    3. All quotes in a single '__unassigned__' bucket.
    """
    membership: dict[str, list[str]] = {}

    # Strategy 1 — direct column
    direct_cols = [c for c in info.columns if "perception" in c.lower()]
    if direct_cols:
        col = direct_cols[0]
        for _, row in info.iterrows():
            gid = str(row.get("global_id", ""))
            perc_val = str(row.get(col, "")).strip()
            if perc_val and perc_val not in ("", "nan"):
                membership.setdefault(perc_val, []).append(gid)
        if membership:
            return membership

    # Strategy 2 — keyword match via thematic areas
    if not perceptions.empty and "topics_thematic_areas" in info.columns:
        perc_keywords: dict[str, list[str]] = {}
        for _, row in perceptions.iterrows():
            pid = str(row.get("global_id", row.get("id", "")))
            text = (
                str(row.get("name", ""))
                + " "
                + str(row.get("quote", ""))
                + " "
                + str(row.get("description", ""))
            ).lower()
            tokens = [t.strip() for t in text.split() if len(t.strip()) > 3]
            if tokens:
                perc_keywords[pid] = tokens

        for _, row in info.iterrows():
            gid = str(row.get("global_id", ""))
            themes = str(row.get("topics_thematic_areas", "")).lower()
            quote_text = str(row.get("quote", "")).lower()
            best_perc = None
            best_score = 0
            for pid, kws in perc_keywords.items():
                score = sum(1 for kw in kws if kw in themes or kw in quote_text)
                if score > best_score:
                    best_score = score
                    best_perc = pid
            target = best_perc if (best_perc and best_score > 0) else "__unassigned__"
            membership.setdefault(target, []).append(gid)
        return membership

    # Strategy 3 — single bucket
    for _, row in info.iterrows():
        membership.setdefault("__unassigned__", []).append(str(row.get("global_id", "")))
    return membership


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------

def compute_diagnostics(
    membership: dict[str, list[str]],
    sem_edges: pd.DataFrame,
    info: pd.DataFrame,
    perceptions: pd.DataFrame,
) -> pd.DataFrame:
    # Build channel lookup
    info_channel: dict[str, str] = {}
    for ch_col in ("channel_id", "channel_code"):
        if ch_col in info.columns:
            info_channel = dict(zip(info["global_id"].astype(str), info[ch_col].astype(str)))
            break

    # Edge lookup: (src, tgt) -> list of {weight, edge_type}
    edge_lookup: dict[tuple[str, str], list[dict]] = {}
    if not sem_edges.empty:
        for _, row in sem_edges.iterrows():
            src = str(row.get("source_global_id", ""))
            tgt = str(row.get("target_global_id", ""))
            w = float(row.get("weight", 0.5))
            etype = str(row.get("edge_type", row.get("connection_type", "")))
            rec = {"weight": w, "edge_type": etype}
            edge_lookup.setdefault((src, tgt), []).append(rec)
            edge_lookup.setdefault((tgt, src), []).append(rec)

    # Perception label map
    perc_label: dict[str, str] = {}
    if not perceptions.empty:
        for _, row in perceptions.iterrows():
            pid = str(row.get("global_id", row.get("id", "")))
            lbl = str(row.get("name", row.get("quote", pid)))[:80]
            perc_label[pid] = lbl

    rows = []
    for perc_id, quote_ids in membership.items():
        quote_set = set(quote_ids)
        n = len(quote_set)
        label = perc_label.get(perc_id, perc_id[:60])

        # Internal coherence and contradiction density
        intra_edges: list[dict] = []
        for i, src in enumerate(quote_ids):
            for tgt in quote_ids[i + 1:]:
                intra_edges.extend(edge_lookup.get((src, tgt), []))

        if intra_edges:
            coherence = round(sum(r["weight"] for r in intra_edges) / len(intra_edges), 4)
            n_contra = sum(1 for r in intra_edges if "contradiction" in r["edge_type"].lower())
            contradiction_density = round(n_contra / len(intra_edges), 4)
        else:
            coherence = None
            contradiction_density = None

        # Purity score
        purity_scores: list[float] = []
        for q in quote_ids:
            all_neighbours = [
                (tgt, max(r["weight"] for r in recs))
                for (src, tgt), recs in edge_lookup.items()
                if src == q
            ]
            top5 = sorted(all_neighbours, key=lambda x: -x[1])[:5]
            if top5:
                n_in = sum(1 for (nb, _) in top5 if nb in quote_set)
                purity_scores.append(n_in / len(top5))
        purity = round(sum(purity_scores) / len(purity_scores), 4) if purity_scores else None

        # Source entropy
        channel_counts: dict[str, int] = {}
        for q in quote_ids:
            ch = info_channel.get(q, "unknown")
            channel_counts[ch] = channel_counts.get(ch, 0) + 1
        source_entropy = round(_shannon_entropy(channel_counts), 4)
        n_channels = len(channel_counts)

        # Status flag
        flags: list[str] = []
        if n < 3:
            flags.append("Underdeveloped (< 3 quotes)")
        if coherence is not None and coherence < 0.40:
            flags.append("Low coherence — candidate for split")
        if source_entropy < 0.5 and n_channels == 1:
            flags.append("Single channel — verify independence")
        if contradiction_density is not None and contradiction_density > 0.30:
            flags.append("High internal contradiction — contested perception")
        if purity is not None and purity < 0.30:
            flags.append("Low purity — quotes closer to other perceptions")
        status = "; ".join(flags) if flags else "Robust"

        rows.append({
            "perception_id": perc_id,
            "perception_label": label,
            "quote_count": n,
            "n_channels": n_channels,
            "internal_coherence": coherence,
            "contradiction_density": contradiction_density,
            "purity_score": purity,
            "source_entropy": source_entropy,
            "status_flag": status,
        })

    return pd.DataFrame(rows).sort_values("quote_count", ascending=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PERCEPTION DIAGNOSTICS — Platform", os.environ.get("KTOOL_PLATFORM_ID", "173"))
    print("=" * 60)

    info, perceptions, sem_edges = load_data()
    print(f"  Quotes loaded:         {len(info)}")
    print(f"  Perceptions loaded:    {len(perceptions)}")
    print(f"  Semantic edges loaded: {len(sem_edges)}")

    membership = build_perception_membership(info, perceptions)
    print(f"\n  Perception groups found: {len(membership)}")
    for pid, qids in membership.items():
        print(f"    {pid[:50]:50s}  {len(qids)} quotes")

    df = compute_diagnostics(membership, sem_edges, info, perceptions)
    write_frame(df, "perception_diagnostics.csv")
    print(f"\n  Saved: perception_diagnostics.csv  ({len(df)} perceptions)")

    display_cols = [c for c in [
        "perception_label", "quote_count", "internal_coherence",
        "purity_score", "source_entropy", "contradiction_density", "status_flag"
    ] if c in df.columns]
    print("\n  Results:")
    print(df[display_cols].to_string(index=False))

    summary = {
        "platform_id": os.environ.get("KTOOL_PLATFORM_ID", "173"),
        "n_perceptions": len(df),
        "total_quotes_assigned": int(df["quote_count"].sum()),
        "perceptions": df[display_cols].fillna("n/a").to_dict(orient="records"),
    }
    write_json(ANALYSIS_DIR / "perception_diagnostics_report.json", summary)
    print("\n  Saved: perception_diagnostics_report.json")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Perception Diagnostics + Narrative Validation — Platform 173.

PHASE 1: Perception Health Metrics (existing)
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

PHASE 2: Narrative Alignment Metrics (new)
  silhouette_score      — (internal_coherence - nearest_other_coherence) /
                          max(...) — how distinct this perception is
  nearest_other         — perception label of the closest other perception
  n_claims              — number of claims extracted from this perception
  claim_vd_profile      — value dimension distribution of claims
  vd_entropy            — Shannon entropy over value dimensions (high = diverse)
  quote_claim_cosim     — mean lexical overlap between perception's quotes and
                          its extracted claims (TF-IDF cosine similarity)
  cross_leakage         — fraction of this perception's top-5 semantic neighbours
                          that belong to OTHER perceptions

PHASE 3: Cross-Perception Differentiation
  js_divergence         — Jensen-Shannon divergence matrix between perception
                          value-dimension profiles
  narratively_redundant — flag when two perceptions have near-identical
                          value profiles (JS < 0.05)

Usage:
  set KTOOL_PLATFORM_ID=173 & set KTOOL_OUTPUT_SUBDIR=test
  python src/analysis/09_perception_diagnostics.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric tokens, 3+ chars."""
    return set(t for t in re.findall(r"[a-zA-Z]\w{2,}", text.lower()))


# ---------------------------------------------------------------------------
# Narrative alignment metrics
# ---------------------------------------------------------------------------

def compute_narrative_alignment(
    membership: dict[str, list[str]],
    sem_edges: pd.DataFrame,
    info: pd.DataFrame,
    perceptions: pd.DataFrame,
    analysis_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute perception-to-narrative-space alignment metrics.

    Returns (perception_profiles, cross_diff) where:
      - perception_profiles: per-perception narrative validation metrics
      - cross_diff: perception × perception value-dimension JS divergence matrix
    """
    # ── Edge lookup: (src, tgt) -> list of weights ──
    edge_lookup: dict[tuple[str, str], list[float]] = {}
    if not sem_edges.empty:
        for _, row in sem_edges.iterrows():
            src = str(row.get("source_global_id", ""))
            tgt = str(row.get("target_global_id", ""))
            w = float(row.get("weight", 0.5))
            edge_lookup.setdefault((src, tgt), []).append(w)
            edge_lookup.setdefault((tgt, src), []).append(w)

    # ── Perception label map ──
    perc_label: dict[str, str] = {}
    if not perceptions.empty:
        for _, row in perceptions.iterrows():
            pid = str(row.get("global_id", row.get("id", "")))
            lbl = str(row.get("name", row.get("quote", pid)))[:80]
            perc_label[pid] = lbl

    # ── Load claim data ──
    claim_nodes_path = analysis_dir / "narrative_layers" / "claim_nodes.csv"
    claim_edges_path = analysis_dir / "narrative_layers" / "claim_edges.csv"
    claim_nodes = pd.DataFrame()
    claim_edges = pd.DataFrame()
    if claim_nodes_path.exists():
        claim_nodes = pd.read_csv(claim_nodes_path)
    if claim_edges_path.exists():
        claim_edges = pd.read_csv(claim_edges_path)

    # ── Build: perception_id -> [claim_ids] ──
    def _normalize_pid(raw: str) -> str:
        return raw.replace("perception_", "").replace("perception", "").strip()

    perc_to_claims: dict[str, list[str]] = {}
    if not claim_edges.empty:
        pmc = claim_edges[claim_edges["edge_type"] == "perception_makes_claim"]
        for _, row in pmc.iterrows():
            src = _normalize_pid(str(row["source_global_id"]))
            tgt = str(row["target_global_id"])
            perc_to_claims.setdefault(src, []).append(tgt)

    # ── Claim text + value dimension lookup ──
    claim_vd: dict[str, str] = {}
    claim_text: dict[str, str] = {}
    if not claim_nodes.empty:
        for _, row in claim_nodes.iterrows():
            gid = str(row["global_id"])
            vd = str(row.get("value_dimension", "") or "")
            if vd and vd != "nan":
                claim_vd[gid] = vd
            txt = str(row.get("description", "") or "")
            if txt and txt != "nan":
                claim_text[gid] = txt
            elif str(row.get("label", "") or "") != "nan":
                claim_text[gid] = str(row.get("label", ""))

    # ── Quote text lookup ──
    quote_text: dict[str, str] = {}
    if not info.empty:
        for _, row in info.iterrows():
            gid = str(row.get("global_id", ""))
            txt = str(row.get("quote", "") or "")
            if txt and txt != "nan":
                quote_text[gid] = txt

    # ── For each perception, compute narrative metrics ──
    vd_by_perc: dict[str, Counter] = {}
    profiles: list[dict] = []

    for perc_id, quote_ids in membership.items():
        qset = set(quote_ids)
        label = perc_label.get(perc_id, perc_id[:60])

        # A. Perception silhouette (distinctness from nearest other)
        #    internal coherence = mean intra-perception edge weight
        intra_weights: list[float] = []
        for i, src in enumerate(quote_ids):
            for tgt in quote_ids[i + 1:]:
                intra_weights.extend(edge_lookup.get((src, tgt), []))
        internal_coherence = np.mean(intra_weights) if intra_weights else 0.0

        #    cross-perception similarity = mean weight to each other perception
        cross_sims: dict[str, list[float]] = {}
        for src in quote_ids:
            for (edge_src, edge_tgt), weights in edge_lookup.items():
                if edge_src == src and edge_tgt not in qset:
                    owner = _owner_of(edge_tgt, membership)
                    if owner:
                        cross_sims.setdefault(owner, []).extend(weights)

        nearest_other = ""
        nearest_sim = 0.0
        for other_pid, sims in cross_sims.items():
            m = np.mean(sims)
            if m > nearest_sim:
                nearest_sim = m
                nearest_other = perc_label.get(other_pid, other_pid[:60])

        silhouette = (internal_coherence - nearest_sim) / max(internal_coherence, nearest_sim, 1e-8) if internal_coherence > 0 else 0.0

        # B. Claims extracted from this perception
        claim_ids = perc_to_claims.get(perc_id, [])
        n_claims = len(claim_ids)

        # C. Value dimension profile
        vd_counter: Counter = Counter()
        for cid in claim_ids:
            vd = claim_vd.get(cid, "")
            if vd:
                vd_counter[vd] += 1
        vd_by_perc[perc_id] = vd_counter
        total_vd = sum(vd_counter.values())
        if total_vd > 0:
            vd_profile = "; ".join(f"{vd}({cnt})" for vd, cnt in vd_counter.most_common())
        else:
            vd_profile = "none"
        vd_list_for_entropy = list(vd_counter.values())
        vd_entropy = _shannon_entropy({str(i): v for i, v in enumerate(vd_list_for_entropy)}) if vd_list_for_entropy else 0.0

        # D. Quote-claim lexical alignment
        perc_texts = [quote_text.get(q, "") for q in quote_ids if q in quote_text]
        claim_texts_for_perc = [claim_text.get(c, "") for c in claim_ids if c in claim_text]
        qc_sim = 0.0
        if perc_texts and claim_texts_for_perc:
            try:
                tfidf_vec = TfidfVectorizer(tokenizer=lambda t: list(_tokenize(t)), max_features=100)
                tfidf_mat = tfidf_vec.fit_transform(perc_texts + claim_texts_for_perc)
                n_q = len(perc_texts)
                q_centroid = np.asarray(tfidf_mat[:n_q].mean(axis=0)).flatten()
                c_centroid = np.asarray(tfidf_mat[n_q:].mean(axis=0)).flatten()
                qc_sim = float(cosine_similarity([q_centroid], [c_centroid])[0, 0])
            except Exception:
                qc_sim = 0.0

        # E. Cross-perception leakage (complement of purity)
        all_leakage_scores: list[float] = []
        for q in quote_ids:
            neighbours = [
                tgt
                for (src, tgt) in edge_lookup
                if src == q and tgt not in qset
            ]
            if neighbours:
                leaked = sum(1 for nb in neighbours if _owner_of(nb, membership) is not None and _owner_of(nb, membership) != perc_id)
                all_leakage_scores.append(leaked / len(neighbours))
        cross_leakage = round(np.mean(all_leakage_scores), 4) if all_leakage_scores else 0.0

        profiles.append({
            "perception_id": perc_id,
            "perception_label": label,
            "n_quotes": len(qset),
            "n_claims": n_claims,
            "internal_coherence_narrative": round(internal_coherence, 4),
            "nearest_other_perception": nearest_other,
            "nearest_other_similarity": round(nearest_sim, 4),
            "silhouette_score": round(silhouette, 4),
            "claim_vd_profile": vd_profile,
            "vd_n_dims": len(vd_counter),
            "vd_entropy": round(abs(vd_entropy), 4),
            "quote_claim_cosim": round(qc_sim, 4),
            "cross_leakage": cross_leakage,
        })

    profile_df = pd.DataFrame(profiles)

    # ── F. Cross-perception claim profile similarity (TF-IDF) ──
    perc_ids = list(vd_by_perc.keys())
    perc_claim_texts: dict[str, str] = {}
    for pid in perc_ids:
        cids = perc_to_claims.get(pid, [])
        texts = [claim_text.get(c, "") for c in cids if c in claim_text]
        perc_claim_texts[pid] = " ".join(texts) if texts else ""

    cross_diff_rows = []
    for i, pid_a in enumerate(perc_ids):
        for j, pid_b in enumerate(perc_ids):
            if i >= j:
                continue
            ta = perc_claim_texts.get(pid_a, "")
            tb = perc_claim_texts.get(pid_b, "")
            claim_sim = 0.0
            if ta and tb:
                try:
                    vec = TfidfVectorizer(tokenizer=lambda t: list(_tokenize(t)), max_features=50)
                    mat = vec.fit_transform([ta, tb])
                    claim_sim = float(cosine_similarity(mat[0:1], mat[1:2])[0, 0])
                except Exception:
                    claim_sim = 0.0
            elif ta == "" and tb == "":
                claim_sim = 1.0  # both have no claims → indistinguishable
            else:
                claim_sim = 0.0  # one has claims, other doesn't → distinguishable

            cross_diff_rows.append({
                "perception_a_id": pid_a,
                "perception_a_label": perc_label.get(pid_a, pid_a[:60]),
                "perception_b_id": pid_b,
                "perception_b_label": perc_label.get(pid_b, pid_b[:60]),
                "claim_cosine_similarity": round(claim_sim, 4),
                "narratively_redundant": claim_sim > 0.80,
            })
    cross_diff_df = pd.DataFrame(cross_diff_rows).sort_values("claim_cosine_similarity", ascending=False)

    return profile_df, cross_diff_df


def _owner_of(global_id: str, membership: dict[str, list[str]]) -> str | None:
    """Return the perception id that owns this global_id, if any."""
    for pid, qids in membership.items():
        if global_id in qids:
            return pid
    return None


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

    # ── PHASE 1: Health diagnostics ──
    df = compute_diagnostics(membership, sem_edges, info, perceptions)
    write_frame(df, "perception_diagnostics.csv")
    print(f"\n  Saved: perception_diagnostics.csv  ({len(df)} perceptions)")

    display_cols = [c for c in [
        "perception_label", "quote_count", "internal_coherence",
        "purity_score", "source_entropy", "contradiction_density", "status_flag"
    ] if c in df.columns]
    print("\n  Results (Phase 1 — Health):")
    print(df[display_cols].to_string(index=False))

    # ── PHASE 2: Narrative alignment ──
    print("\n" + "=" * 60)
    print("PHASE 2: NARRATIVE ALIGNMENT")
    print("=" * 60)
    profile_df, cross_diff_df = compute_narrative_alignment(membership, sem_edges, info, perceptions, ANALYSIS_DIR)

    write_frame(profile_df, "perception_narrative_profiles.csv")
    print(f"  Saved: perception_narrative_profiles.csv  ({len(profile_df)} perceptions)")

    nalign_cols = [c for c in [
        "perception_label", "n_quotes", "n_claims",
        "silhouette_score", "nearest_other_perception", "nearest_other_similarity",
        "vd_n_dims", "vd_entropy", "quote_claim_cosim", "cross_leakage",
    ] if c in profile_df.columns]
    print("\n  Narrative alignment results:")
    print(profile_df[nalign_cols].to_string(index=False))

    if not cross_diff_df.empty:
        write_frame(cross_diff_df, "perception_value_divergence.csv")
        red = cross_diff_df[cross_diff_df["narratively_redundant"]]
        if len(red) > 0:
            print(f"\n  ⚠️ Narratively redundant pairs (claim cosine similarity > 0.80):")
            for _, row in red.iterrows():
                print(f"    {row['perception_a_label']:20s} ↔ {row['perception_b_label']:20s}  sim={row['claim_cosine_similarity']:.4f}")
        else:
            print(f"\n  ✅ No narratively redundant pairs — all perceptions have distinct claim profiles.")

    # ── Summary report ──
    summary = {
        "platform_id": os.environ.get("KTOOL_PLATFORM_ID", "173"),
        "n_perceptions": len(df),
        "total_quotes_assigned": int(df["quote_count"].sum()),
        "perceptions": df[display_cols].fillna("n/a").to_dict(orient="records"),
        "narrative_validation": profile_df[nalign_cols].fillna("n/a").to_dict(orient="records") if not profile_df.empty else [],
        "narratively_redundant_pairs": len(red) if not cross_diff_df.empty else 0,
        "perception_value_divergence": cross_diff_df.fillna("n/a").to_dict(orient="records") if not cross_diff_df.empty else [],
    }
    write_json(ANALYSIS_DIR / "perception_diagnostics_report.json", summary)
    print("\n  Saved: perception_diagnostics_report.json")
    print("=" * 60)


if __name__ == "__main__":
    main()

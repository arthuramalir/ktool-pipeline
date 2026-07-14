from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

from graph_utils import ANALYSIS_DIR, ANALYTICS_DIR, DATA_DIR, PLATFORM_ID, load_table

VD_ORDER = [
    "cultural_identity",
    "social_justice",
    "collaboration",
    "innovation_drive",
    "evidence_based",
    "community_autonomy",
    "austerity_scarcity",
]
M = len(VD_ORDER)
VD_TO_IDX = {v: i for i, v in enumerate(VD_ORDER)}

BELIEF_WEIGHT = {"deep_core": 1.0, "secondary": 0.6, "surface": 0.3}

EPSILON_DEFAULT = 0.8
MAX_ITER = 300
TOL = 1e-5
BETA = 0.05


def cosine_sim_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    return (mat / norms) @ (mat / norms).T


class GBCMFJSimulator:
    def __init__(
        self,
        adj_matrix: np.ndarray,
        node_ids: list[str],
        embeddings: np.ndarray,
        initial_opinions: np.ndarray,
        is_claim: np.ndarray,
        is_perception: np.ndarray,
        perception_n_quotes: dict[str, int],
        value_correlation: np.ndarray | None = None,
        beta: float = BETA,
    ):
        self.N = len(node_ids)
        self.M = initial_opinions.shape[1]
        self.node_ids = node_ids
        self.id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        self.X0 = torch.tensor(initial_opinions, dtype=torch.float32)

        self.A = torch.tensor(adj_matrix, dtype=torch.float32)

        sim_mat = cosine_sim_rows(embeddings)
        sim_mat = np.clip(sim_mat, 0.0, 1.0)
        self.Sim = torch.tensor(sim_mat, dtype=torch.float32)

        if value_correlation is not None:
            self.C = torch.tensor(value_correlation, dtype=torch.float32)
        else:
            self.C = torch.eye(self.M, dtype=torch.float32)

        lam_diag = np.ones(self.N, dtype=np.float32)
        for i, nid in enumerate(node_ids):
            if is_claim[i]:
                lam_diag[i] = 0.0
            elif is_perception[i]:
                nq = perception_n_quotes.get(nid, 1)
                alpha = 1.0 - np.exp(-beta * nq)
                lam_diag[i] = 1.0 - alpha
        self.Lambda = torch.tensor(np.diag(lam_diag), dtype=torch.float32)
        self.I_minus_Lambda_X0 = torch.tensor(
            np.diag(1.0 - lam_diag) @ self.X0.numpy(), dtype=torch.float32
        )

    def _transition_matrix(self, A_mat: torch.Tensor, X: torch.Tensor, epsilon: float) -> torch.Tensor:
        diff = X.unsqueeze(1) - X.unsqueeze(0)
        diff_t = torch.matmul(diff, self.C)
        discordance = torch.sqrt(torch.sum(diff * diff_t, dim=2).clamp(min=0.0))
        gate = (discordance <= epsilon).float()
        W_raw = A_mat * self.Sim * gate
        row_sums = torch.sum(W_raw, dim=1, keepdim=True)
        W = W_raw / row_sums.clamp(min=1e-8)
        isolated = (row_sums.squeeze() == 0).float()
        W = W + torch.diag(isolated)
        return W

    def simulate(
        self, counterfactual_edges: list[tuple[str, str]] | None = None, epsilon: float = EPSILON_DEFAULT
    ) -> tuple[np.ndarray, np.ndarray, int]:
        A_sim = self.A.clone()
        if counterfactual_edges:
            for u, v in counterfactual_edges:
                iu = self.id_to_idx.get(u)
                iv = self.id_to_idx.get(v)
                if iu is not None and iv is not None:
                    A_sim[iu, iv] = 1.0
                    A_sim[iv, iu] = 1.0
        X = self.X0.clone()
        n_iters = 0
        for _ in range(MAX_ITER):
            X_prev = X.clone()
            W = self._transition_matrix(A_sim, X, epsilon)
            X_next = torch.mm(torch.mm(self.Lambda, W), X)
            X_next = torch.matmul(X_next, self.C.t()) + self.I_minus_Lambda_X0
            delta = torch.norm(X_next - X, p="fro")
            X = X_next
            n_iters += 1
            if delta.item() < TOL:
                break
        return X.numpy(), W.numpy(), n_iters

    def cold_start_node(
        self,
        label: str,
        description: str,
        embedding: np.ndarray,
        value_dimension: str | None = None,
        existing_embeddings: np.ndarray | None = None,
        threshold: float = 0.5,
        top_k: int = 5,
    ) -> tuple[np.ndarray, list[str], list[float]]:
        if existing_embeddings is None:
            existing_embeddings = self.Sim.numpy()
            norms = np.linalg.norm(existing_embeddings, axis=1, keepdims=True)
            existing_embeddings = existing_embeddings / np.where(norms == 0, 1e-8, norms)
        emb_norm = embedding / max(np.linalg.norm(embedding), 1e-8)
        sims = existing_embeddings @ emb_norm
        sims = np.clip(sims, 0.0, 1.0)
        above = np.where(sims >= threshold)[0]
        if len(above) < top_k:
            above = np.argsort(-sims)[:top_k]
        pred_edges = [(self.node_ids[i], sims[i]) for i in above]
        x0 = np.zeros(self.M, dtype=np.float32)
        if value_dimension and value_dimension in VD_TO_IDX:
            x0[VD_TO_IDX[value_dimension]] = 1.0
        conn_vec = np.zeros(self.N, dtype=np.float32)
        for i, _ in pred_edges:
            conn_vec[self.id_to_idx[i]] = 1.0
        return x0, [e[0] for e in pred_edges], [float(e[1]) for e in pred_edges]


def build_initial_opinions(nodes_df: pd.DataFrame, perception_claim_map: dict[str, list[int]] | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    N = len(nodes_df)
    prior = 1.0 / M
    X0 = np.full((N, M), prior, dtype=np.float32)
    claim_X0 = {}
    for i, row in nodes_df.iterrows():
        gid = str(row.get("global_id", ""))
        nt = str(row.get("node_type", ""))
        if nt == "claim":
            vd = str(row.get("value_dimension", "")).strip()
            bl = str(row.get("belief_level", "")).strip().lower()
            if vd in VD_TO_IDX:
                w = BELIEF_WEIGHT.get(bl, 0.5)
                X0[i] = np.array([0.0] * M, dtype=np.float32)
                X0[i, VD_TO_IDX[vd]] = w
                claim_X0[gid] = X0[i].copy()
        elif nt == "perception" and perception_claim_map and gid in perception_claim_map:
            dims = perception_claim_map[gid]
            if dims:
                v = np.zeros(M, dtype=np.float32)
                for d in dims:
                    v[d] += 1.0
                v = v / v.sum()
                X0[i] = v

    return X0, claim_X0


def build_value_correlation(nodes_df: pd.DataFrame) -> np.ndarray:
    return np.eye(M, dtype=np.float32)


def build_graph_data() -> (
    tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, int]]
):
    nodes = load_table("nodes.csv", "analytics")
    edges = load_table("edges.csv", "analytics")
    claim_nodes = load_table("claim_nodes.csv", "narrative_layers")
    perception_profiles = load_table("perception_narrative_profiles.csv", "analysis")

    node_ids = nodes["global_id"].astype(str).tolist()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    N = len(node_ids)

    claim_vd_map: dict[str, int] = {}
    for _, row in claim_nodes.iterrows():
        gid = str(row.get("global_id", ""))
        vd = str(row.get("value_dimension", "")).strip()
        if vd in VD_TO_IDX:
            claim_vd_map[gid] = VD_TO_IDX[vd]

    perception_claim_map: dict[str, list[int]] = {}
    claim_edge_path = ANALYSIS_DIR / "narrative_layers" / "claim_edges.csv"
    if claim_edge_path.exists():
        cedf = pd.read_csv(claim_edge_path)
        for _, row in cedf.iterrows():
            src = str(row.get("source_global_id", ""))
            tgt = str(row.get("target_global_id", ""))
            if row.get("edge_type") == "perception_makes_claim" and tgt in claim_vd_map:
                perception_claim_map.setdefault(src, []).append(claim_vd_map[tgt])

    emb_dim = 384
    embeddings = np.zeros((N, emb_dim), dtype=np.float32)

    cache_path = ANALYTICS_DIR / "gbcm_fj_embeddings_cache.npy" if ANALYTICS_DIR else ANALYSIS_DIR / "gbcm_fj_embeddings_cache.npy"
    if cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path.name}")
        cached = np.load(cache_path)
        if cached.shape[0] == N:
            embeddings = cached
            desc_indices = []
        else:
            print(f"  Cache shape mismatch ({cached.shape[0]} vs {N}), recomputing")
            desc_indices = list(range(N))
    else:
        desc_indices = list(range(N))

    if desc_indices:
        claim_desc = {}
        for _, row in claim_nodes.iterrows():
            gid = str(row.get("global_id", ""))
            desc = str(row.get("description", "")).strip()
            if desc:
                claim_desc[gid] = desc

        desc_texts = []
        valid_indices = []
        for i in desc_indices:
            row = nodes.iloc[i]
            gid = str(row.get("global_id", ""))
            desc = ""
            for col in ["description", "quote", "name"]:
                val = str(row.get(col, "")).strip()
                if val and val.lower() not in ("nan", "", "none"):
                    desc = val
                    break
            if gid in claim_desc:
                desc = claim_desc[gid]
            if desc:
                desc_texts.append(desc)
                valid_indices.append(i)

        if desc_texts:
            print(f"  Encoding {len(desc_texts)} descriptions with SentenceTransformer...")
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            vecs = model.encode(desc_texts, show_progress_bar=True, convert_to_numpy=True)
            for idx, vec in zip(valid_indices, vecs):
                embeddings[idx] = vec.astype(np.float32)
            np.save(cache_path, embeddings)
            print(f"  Cached embeddings to {cache_path.name}")

    print(f"  Building adjacency matrix...")
    adj = np.zeros((N, N), dtype=np.float32)
    for _, row in edges.iterrows():
        src = str(row.get("source_global_id", ""))
        tgt = str(row.get("target_global_id", ""))
        i_src = id_to_idx.get(src)
        i_tgt = id_to_idx.get(tgt)
        if i_src is not None and i_tgt is not None:
            adj[i_src, i_tgt] = 1.0
            adj[i_tgt, i_src] = 1.0

    node_types = nodes["node_type"].astype(str).str.lower().tolist()
    is_claim = np.array([t == "claim" for t in node_types], dtype=bool)
    is_perception = np.array([t == "perception" for t in node_types], dtype=bool)

    perception_quotes = {}
    for _, row in perception_profiles.iterrows():
        pid = str(row.get("perception_id", ""))
        nq = int(row.get("n_quotes", 0))
        for gid in node_ids:
            if gid.endswith(f"_{pid}") or gid == f"perception_{pid}":
                perception_quotes[gid] = nq
                break
    for i, nid in enumerate(node_ids):
        if is_perception[i] and nid not in perception_quotes:
            perception_quotes[nid] = 1

    print(f"  Building initial opinions and value correlation...")
    X0, _ = build_initial_opinions(nodes, perception_claim_map)
    C_mat = build_value_correlation(nodes)
    return adj, node_ids, embeddings, X0, is_claim, is_perception, perception_quotes, C_mat


def compute_perception_pagerank(W: np.ndarray, perc_indices: np.ndarray, damping: float = 0.85, max_iter: int = 100) -> np.ndarray:
    N = W.shape[0]
    pr = np.ones(N) / N
    teleport = (1.0 - damping) / N
    for _ in range(max_iter):
        new_pr = teleport + damping * (pr @ W)
        if np.abs(new_pr - pr).sum() < 1e-8:
            break
        pr = new_pr
    v = pr / pr.sum()
    return v


def compute_disagreement(W: np.ndarray, X: np.ndarray) -> float:
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    return float(np.sum(W * d))


def compute_exposure(perc_idx: int, W: np.ndarray, X: np.ndarray) -> float:
    return float(W[perc_idx, :] @ X.mean(axis=1))


def compute_metrics(
    X_baseline: np.ndarray,
    X_intervention: np.ndarray,
    W_baseline: np.ndarray,
    W_intervention: np.ndarray,
    node_ids: list[str],
    is_perception: np.ndarray,
    is_claim: np.ndarray,
) -> dict:
    metrics = {}

    perc_indices = np.where(is_perception)[0]

    v = compute_perception_pagerank(W_intervention, perc_indices)
    v_b = compute_perception_pagerank(W_baseline, perc_indices)

    for pi in perc_indices:
        nid = node_ids[pi]
        delta = v[pi] - v_b[pi]
        metrics[f"pr_delta_{nid}"] = float(delta)

    for pi in perc_indices:
        nid = node_ids[pi]
        metrics[f"exposure_{nid}"] = compute_exposure(pi, W_intervention, X_intervention)

    metrics["disagreement_baseline"] = compute_disagreement(W_baseline, X_baseline)
    metrics["disagreement_intervention"] = compute_disagreement(W_intervention, X_intervention)

    mean_opinion = X_intervention.mean(axis=0)
    p = mean_opinion / mean_opinion.sum()
    hill_d1 = np.exp(-np.sum(p * np.log(p + 1e-10)))
    metrics["hill_diversity_d1"] = float(hill_d1)

    return metrics


def main() -> None:
    print("=== GBCM-FJ Narrative-Opinion Simulation ===")

    adj, node_ids, embeddings, X0, is_claim, is_perception, perception_quotes, C_mat = build_graph_data()

    sim = GBCMFJSimulator(
        adj_matrix=adj,
        node_ids=node_ids,
        embeddings=embeddings,
        initial_opinions=X0,
        is_claim=is_claim,
        is_perception=is_perception,
        perception_n_quotes=perception_quotes,
        value_correlation=C_mat,
    )

    print(f"  Graph: {sim.N} nodes, {sim.M} value dimensions")
    print(f"  Claims: {is_claim.sum()}, Perceptions: {is_perception.sum()}")
    print(f"  Baseline simulation...")
    X_base, W_base, n_iter = sim.simulate(epsilon=EPSILON_DEFAULT)
    print(f"  Converged in {n_iter} iterations")

    baseline_state_path = ANALYSIS_DIR / "gbcm_fj_baseline_state.npy"
    baseline_W_path = ANALYSIS_DIR / "gbcm_fj_baseline_W.npy"
    np.save(baseline_state_path, X_base)
    np.save(baseline_W_path, W_base)
    print(f"  Baseline state saved: {baseline_state_path.name}")

    cand_path = ANALYSIS_DIR / "narrative_impact_predictions.csv"
    if not cand_path.exists():
        print("  No candidate edges — skipping intervention simulation")
        return

    candidates = pd.read_csv(cand_path)
    rows = []
    for _, cand in candidates.iterrows():
        src = str(cand.get("source_global_id", ""))
        tgt = str(cand.get("target_global_id", ""))
        if src not in sim.id_to_idx or tgt not in sim.id_to_idx:
            continue
        print(f"  Simulating: {src} <-> {tgt}")
        X_int, W_int, n_int = sim.simulate(
            counterfactual_edges=[(src, tgt)], epsilon=EPSILON_DEFAULT
        )
        metrics = compute_metrics(X_base, X_int, W_base, W_int, node_ids, is_perception, is_claim)
        row = {
            "source_global_id": src,
            "target_global_id": tgt,
            "converged_iterations": n_int,
            "hill_diversity_d1_baseline": metrics.get("hill_diversity_d1", 0.0),
            "hill_diversity_d1_intervention": metrics.get("hill_diversity_d1", 0.0),
            "disagreement_baseline": metrics.get("disagreement_baseline", 0.0),
            "disagreement_intervention": metrics.get("disagreement_intervention", 0.0),
        }
        for k, v in metrics.items():
            if k.startswith("pr_delta_") or k.startswith("exposure_"):
                row[k] = v
        rows.append(row)

    results_path = ANALYSIS_DIR / "gbcm_fj_intervention_results.csv"
    if rows:
        pd.DataFrame(rows).to_csv(results_path, index=False)
        print(f"  {len(rows)} intervention results saved: {results_path.name}")

    perc_indices = np.where(is_perception)[0]
    perc_rows = []
    for pi in perc_indices:
        nid = node_ids[pi]
        perc_rows.append(
            {
                "perception_id": nid,
                "perception_index": int(pi),
                "baseline_opinion": json.dumps(X_base[pi].tolist()),
            }
        )
    if perc_rows:
        perc_path = ANALYSIS_DIR / "gbcm_fj_perception_baseline.csv"
        pd.DataFrame(perc_rows).to_csv(perc_path, index=False)
        print(f"  Perception baseline saved: {perc_path.name}")

    summary = {
        "n_nodes": sim.N,
        "n_value_dimensions": sim.M,
        "n_claims": int(is_claim.sum()),
        "n_perceptions": int(is_perception.sum()),
        "epsilon": EPSILON_DEFAULT,
        "beta": BETA,
        "baseline_converged_iterations": n_iter,
        "interventions_simulated": len(rows),
    }
    summary_path = ANALYSIS_DIR / "gbcm_fj_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {summary_path.name}")
    print("=== GBCM-FJ simulation complete ===")


if __name__ == "__main__":
    main()

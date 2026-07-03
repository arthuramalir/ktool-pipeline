from __future__ import annotations

import argparse
import json
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173_synthetic")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
DEFAULT_GNN_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR / "analysis" / "gnn"
MAPPING_LAYER_TYPES = {"agent", "project"}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def require_torch_packages():
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch_geometric.nn import GCNConv
        return torch, nn, F, GCNConv
    except ImportError as exc:  # pragma: no cover - graceful runtime guard
        raise SystemExit(
            "PyTorch Geometric link prediction requires `torch` and `torch_geometric`. "
            "Install them first, then rerun this script."
        ) from exc


def discover_gnn_dir(preferred: Path) -> Path:
    candidates = [
        preferred,
        ROOT / "data" / "processed" / "173_synthetic" / OUTPUT_SUBDIR / "analysis" / "gnn",
        ROOT / "data" / "processed" / "173" / OUTPUT_SUBDIR / "analysis" / "gnn",
    ]
    for candidate in candidates:
        if (candidate / "gnn_summary.json").exists():
            return candidate
    return preferred


def load_artifacts(gnn_dir: Path) -> dict[str, Any]:
    summary = load_json(gnn_dir / "gnn_summary.json")
    required = {
        "node_features": gnn_dir / "node_features.npy",
        "edge_index": gnn_dir / "edge_index.npy",
        "edge_weight": gnn_dir / "edge_weight.npy",
        "edge_attr": gnn_dir / "edge_attr.npy",
        "node_index": gnn_dir / "node_index.csv",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing GNN artifacts: {', '.join(missing)} in {gnn_dir}")

    return {
        "summary": summary,
        "x": np.load(required["node_features"]),
        "edge_index": np.load(required["edge_index"]),
        "edge_weight": np.load(required["edge_weight"]),
        "edge_attr": np.load(required["edge_attr"]),
        "node_index": pd.read_csv(required["node_index"]),
    }


def split_edges(edge_index: np.ndarray, edge_weight: np.ndarray, seed: int, train_ratio: float, val_ratio: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_edges = edge_index.shape[1]
    indices = np.arange(n_edges)
    rng.shuffle(indices)
    train_count = max(1, int(round(n_edges * train_ratio)))
    val_count = max(1, int(round(n_edges * val_ratio)))
    if train_count + val_count >= n_edges:
        val_count = max(1, min(val_count, n_edges - train_count - 1))
    train_end = train_count
    val_end = min(n_edges, train_end + val_count)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "train_pos": edge_index[:, train_idx],
        "val_pos": edge_index[:, val_idx],
        "test_pos": edge_index[:, test_idx],
        "train_weight": edge_weight[train_idx],
        "val_weight": edge_weight[val_idx],
        "test_weight": edge_weight[test_idx],
    }


def edge_key(source: int, target: int) -> tuple[int, int]:
    return int(source), int(target)


def sample_negative_edges(num_samples: int, n_nodes: int, positive_set: set[tuple[int, int]], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    negatives: list[tuple[int, int]] = []
    seen = set()
    max_attempts = max(num_samples * 40, 1000)
    attempts = 0
    while len(negatives) < num_samples and attempts < max_attempts:
        batch = min(max(4, num_samples - len(negatives)), 256)
        sources = rng.integers(0, n_nodes, size=batch)
        targets = rng.integers(0, n_nodes, size=batch)
        for source, target in zip(sources, targets, strict=False):
            attempts += 1
            pair = edge_key(source, target)
            if source == target or pair in positive_set or pair in seen:
                continue
            seen.add(pair)
            negatives.append(pair)
            if len(negatives) >= num_samples:
                break
    if len(negatives) < num_samples:
        raise RuntimeError("Unable to sample enough negative edges for link prediction.")
    return np.asarray(negatives, dtype=np.int64).T


def build_encoder(nn, GCNConv, input_dim: int, hidden_dim: int):
    class LinkEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = nn.functional.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            return x

    return LinkEncoder()


def decode_dot(z, edge_index):
    source = z[edge_index[0]]
    target = z[edge_index[1]]
    return (source * target).sum(dim=-1)


def recommendation_category(source_type: str, target_type: str) -> tuple[str, str, str]:
    pair = {source_type, target_type}
    if pair <= {"project"}:
        return "Project alignment", "Structure", "Likely to reinforce collaboration between initiatives in the mapping layer."
    if pair <= {"agent"}:
        return "Actor coordination", "Structure", "Likely to strengthen coordination between actors in the mapping layer."
    if pair == {"agent", "project"}:
        return "Actor-initiative bridge", "Structure", "Likely to improve how actors and initiatives connect in the mapping layer."
    if pair <= {"pilot", "prototype"}:
        return "Experiment alignment", "Structure", "Likely to connect experimental initiatives in the mapping layer."
    if pair <= {"agent", "pilot"} or pair <= {"project", "pilot"}:
        return "Delivery bridge", "Structure", "Likely to connect a mapping-layer actor with an initiative under test."
    return "Bridge strengthening", "Structure", "Likely to connect two weakly linked parts of the mapping layer."


def link_type_label(source_type: str, target_type: str) -> str:
    return f"{source_type} -> {target_type}"


def pair_kind_label(source_type: str, target_type: str) -> str:
    if source_type == target_type:
        return f"{source_type}-{target_type}"
    if {source_type, target_type} == {"agent", "project"}:
        return "project-agent"
    return f"{source_type}-{target_type}"


def build_neighbors(edge_index: np.ndarray, n_nodes: int) -> list[set[int]]:
    neighbors = [set() for _ in range(n_nodes)]
    for source, target in edge_index.T.tolist():
        source_id = int(source)
        target_id = int(target)
        if source_id == target_id:
            continue
        neighbors[source_id].add(target_id)
        neighbors[target_id].add(source_id)
    return neighbors


def collect_perception_nodes(start: int, neighbors: list[set[int]], node_types: list[str], max_hops: int = 3) -> set[int]:
    seen = {start}
    frontier = {start}
    perceptions: set[int] = set()
    for _ in range(max_hops):
        next_frontier: set[int] = set()
        for node in frontier:
            for neighbor in neighbors[node]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                next_frontier.add(neighbor)
                if node_types[neighbor] == "perception":
                    perceptions.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return perceptions


def format_node_labels(node_ids: set[int], node_index: pd.DataFrame, limit: int = 4) -> str:
    if not node_ids:
        return "None"
    labels = [public_label(node_index.iloc[int(node_id)]) for node_id in sorted(node_ids)]
    if len(labels) > limit:
        labels = labels[:limit] + [f"+{len(node_ids) - limit} more"]
    return "; ".join(labels)


def perception_effect_summary(
    source_index: int,
    target_index: int,
    neighbors: list[set[int]],
    node_types: list[str],
    node_index: pd.DataFrame,
) -> dict[str, Any]:
    source_perceptions = collect_perception_nodes(source_index, neighbors, node_types, max_hops=3)
    target_perceptions = collect_perception_nodes(target_index, neighbors, node_types, max_hops=3)
    shared_perceptions = source_perceptions.intersection(target_perceptions)
    union_perceptions = source_perceptions.union(target_perceptions)

    if shared_perceptions:
        effect_type = "Reinforces shared perception space"
        effect_note = "The proposed mapping link sits near the same perception neighborhoods on both sides, so it may strengthen an already shared narrative frame."
    elif source_perceptions and target_perceptions:
        effect_type = "Bridges separate perception spaces"
        effect_note = "The proposed mapping link connects two different perception neighborhoods, so it may create a new bridge between narrative frames."
    elif source_perceptions or target_perceptions:
        effect_type = "Extends perception reach"
        effect_note = "Only one endpoint currently touches perception neighborhoods, so the proposed link may extend that narrative influence to a new actor or project."
    else:
        effect_type = "No immediate perception footprint"
        effect_note = "Neither endpoint currently sits near perception nodes, so the proposed link is mainly a structural mapping link rather than a near-term perception-space change."

    union_size = len(union_perceptions)
    shared_size = len(shared_perceptions)
    overlap_ratio = float(shared_size / union_size) if union_size else 0.0
    bridge_score = float(union_size + shared_size * 0.5)

    return {
        "source_perception_count": int(len(source_perceptions)),
        "target_perception_count": int(len(target_perceptions)),
        "shared_perception_count": int(shared_size),
        "union_perception_count": int(union_size),
        "perception_overlap_ratio": overlap_ratio,
        "perception_bridge_score": bridge_score,
        "perception_effect_type": effect_type,
        "perception_effect_note": effect_note,
        "source_perceptions": format_node_labels(source_perceptions, node_index),
        "target_perceptions": format_node_labels(target_perceptions, node_index),
        "shared_perceptions": format_node_labels(shared_perceptions, node_index),
    }


def weighted_link_loss(z, pos_edge_index, pos_edge_weight, neg_edge_index, torch, F):
    pos_logits = decode_dot(z, pos_edge_index)
    neg_logits = decode_dot(z, neg_edge_index)
    pos_target = torch.ones(pos_logits.size(0), device=z.device)
    neg_target = torch.zeros(neg_logits.size(0), device=z.device)
    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_target, reduction="none")
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_target, reduction="none")
    pos_weights = pos_edge_weight.to(z.device)
    pos_weights = pos_weights / max(float(pos_weights.mean().item()), 1.0e-6)
    return (pos_loss * pos_weights).mean() + neg_loss.mean()


def score_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    if y_true.size == 0:
        return {"auc": 0.0, "average_precision": 0.0}
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)
    if pos == 0 or neg == 0:
        return {"auc": 0.0, "average_precision": 0.0}
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_ranks = ranks[y_true == 1]
    auc = float((pos_ranks.sum() - pos * (pos + 1) / 2.0) / (pos * neg))
    sorted_idx = np.argsort(-y_score)
    sorted_true = y_true[sorted_idx]
    precision_sum = 0.0
    hit_count = 0
    for rank, label in enumerate(sorted_true, start=1):
        if label == 1:
            hit_count += 1
            precision_sum += hit_count / rank
    average_precision = float(precision_sum / pos)
    return {"auc": auc, "average_precision": average_precision}


def public_label(row: pd.Series) -> str:
    label = str(row.get("public_label", "")).strip()
    if label and label.lower() not in {"unnamed", "nan", "none"}:
        return label
    label = str(row.get("label", "")).strip()
    if label and label.lower() not in {"unnamed", "nan", "none"}:
        return label
    global_id = str(row.get("global_id", "")).strip()
    return global_id or "Unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small GAE-style link predictor on the prepared ALC K-Tool GNN tensors")
    parser.add_argument("--gnn-dir", type=Path, default=DEFAULT_GNN_DIR)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-predictions", type=int, default=50)
    args = parser.parse_args()

    torch, nn, F, GCNConv = require_torch_packages()
    gnn_dir = discover_gnn_dir(args.gnn_dir)
    artifacts = load_artifacts(gnn_dir)
    summary = artifacts["summary"]

    x = torch.tensor(artifacts["x"], dtype=torch.float32)
    edge_index = artifacts["edge_index"].astype(np.int64)
    edge_weight = artifacts["edge_weight"].astype(np.float32)
    node_index = artifacts["node_index"].copy()
    node_types = node_index["node_type"].astype(str).str.lower().tolist()
    neighbors = build_neighbors(edge_index, x.shape[0])

    keep_mask = edge_weight >= float(args.confidence_threshold)
    if keep_mask.any():
        edge_index = edge_index[:, keep_mask]
        edge_weight = edge_weight[keep_mask]

    if edge_index.shape[1] < 10:
        raise ValueError("Not enough edges remain after confidence filtering to train a link predictor.")

    split = split_edges(edge_index, edge_weight, args.seed, args.train_ratio, args.val_ratio)
    n_nodes = x.shape[0]
    positive_set = {edge_key(int(s), int(t)) for s, t in split["train_pos"].T.tolist()}
    positive_set.update({edge_key(int(s), int(t)) for s, t in split["val_pos"].T.tolist()})
    positive_set.update({edge_key(int(s), int(t)) for s, t in split["test_pos"].T.tolist()})

    train_neg = sample_negative_edges(split["train_pos"].shape[1], n_nodes, positive_set, args.seed + 1)
    val_neg = sample_negative_edges(split["val_pos"].shape[1], n_nodes, positive_set, args.seed + 2)
    test_neg = sample_negative_edges(split["test_pos"].shape[1], n_nodes, positive_set, args.seed + 3)

    train_pos = torch.tensor(split["train_pos"], dtype=torch.long)
    val_pos = torch.tensor(split["val_pos"], dtype=torch.long)
    test_pos = torch.tensor(split["test_pos"], dtype=torch.long)
    train_neg = torch.tensor(train_neg, dtype=torch.long)
    val_neg = torch.tensor(val_neg, dtype=torch.long)
    test_neg = torch.tensor(test_neg, dtype=torch.long)
    train_weight = torch.tensor(split["train_weight"], dtype=torch.float32)

    model = build_encoder(nn, GCNConv, x.shape[1], args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_auc = -1.0
    history = []
    patience = 20
    patience_left = patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model(x, train_pos)
        loss = weighted_link_loss(z, train_pos, train_weight, train_neg, torch, F)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z = model(x, train_pos)
            train_scores = torch.sigmoid(decode_dot(z, train_pos)).cpu().numpy()
            train_labels = np.ones(train_pos.shape[1], dtype=np.int64)
            train_neg_scores = torch.sigmoid(decode_dot(z, train_neg)).cpu().numpy()
            train_labels = np.concatenate([train_labels, np.zeros(train_neg.shape[1], dtype=np.int64)])
            train_scores = np.concatenate([train_scores, train_neg_scores])

            val_pos_scores = torch.sigmoid(decode_dot(z, val_pos)).cpu().numpy()
            val_neg_scores = torch.sigmoid(decode_dot(z, val_neg)).cpu().numpy()
            val_labels = np.concatenate([
                np.ones(val_pos.shape[1], dtype=np.int64),
                np.zeros(val_neg.shape[1], dtype=np.int64),
            ])
            val_scores = np.concatenate([val_pos_scores, val_neg_scores])

            test_pos_scores = torch.sigmoid(decode_dot(z, test_pos)).cpu().numpy()
            test_neg_scores = torch.sigmoid(decode_dot(z, test_neg)).cpu().numpy()
            test_labels = np.concatenate([
                np.ones(test_pos.shape[1], dtype=np.int64),
                np.zeros(test_neg.shape[1], dtype=np.int64),
            ])
            test_scores = np.concatenate([test_pos_scores, test_neg_scores])

            train_metrics = score_metrics(train_labels, train_scores)
            val_metrics = score_metrics(val_labels, val_scores)
            test_metrics = score_metrics(test_labels, test_scores)

        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "train_auc": train_metrics["auc"],
                "train_average_precision": train_metrics["average_precision"],
                "validation_auc": val_metrics["auc"],
                "validation_average_precision": val_metrics["average_precision"],
                "test_auc": test_metrics["auc"],
                "test_average_precision": test_metrics["average_precision"],
            }
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        z = model(x, train_pos)

    candidate_scores = []
    n_candidate = int(args.top_k_predictions)
    all_positive = positive_set.copy()
    rng = np.random.default_rng(args.seed + 99)
    candidate_neg = sample_negative_edges(max(n_candidate * 6, n_candidate), n_nodes, all_positive, args.seed + 99)
    candidate_scores_tensor = torch.sigmoid(decode_dot(z, torch.tensor(candidate_neg, dtype=torch.long))).cpu().numpy()
    candidate_pairs = candidate_neg.T.tolist()
    ranked_candidates = sorted(zip(candidate_pairs, candidate_scores_tensor, strict=False), key=lambda item: item[1], reverse=True)
    for (source, target), score in ranked_candidates:
        source_row = node_index.iloc[int(source)]
        target_row = node_index.iloc[int(target)]
        source_type = str(source_row.get("node_type", "unknown"))
        target_type = str(target_row.get("node_type", "unknown"))
        if source_type not in MAPPING_LAYER_TYPES or target_type not in MAPPING_LAYER_TYPES:
            continue
        category, impact_layer, rationale = recommendation_category(source_type, target_type)
        candidate_scores.append(
            {
                "source_index": int(source),
                "target_index": int(target),
                "source_global_id": str(node_index.iloc[int(source)]["global_id"]),
                "target_global_id": str(node_index.iloc[int(target)]["global_id"]),
                "source_label": public_label(source_row),
                "target_label": public_label(target_row),
                "source_node_type": source_type,
                "target_node_type": target_type,
                "link_type": link_type_label(source_type, target_type),
                "pair_kind": pair_kind_label(source_type, target_type),
                "recommendation_category": category,
                "impact_layer": impact_layer,
                "rationale": rationale,
                "link_probability": float(score),
            }
        )
        if len(candidate_scores) >= n_candidate:
            break

    candidate_frame = pd.DataFrame(candidate_scores)
    shortlist_rows = []
    if not candidate_frame.empty:
        shortlist_frame = candidate_frame.copy()
        priority_pair_kinds = ["project-project", "agent-agent", "project-agent", "agent-project"]
        for pair_kind in priority_pair_kinds:
            subset = shortlist_frame[shortlist_frame["pair_kind"] == pair_kind].sort_values("link_probability", ascending=False)
            if subset.empty:
                continue
            row = subset.iloc[0]
            shortlist_rows.append(row.to_dict())
        if len(shortlist_rows) < 3:
            remainder = shortlist_frame.sort_values("link_probability", ascending=False)
            for _, row in remainder.iterrows():
                row_dict = row.to_dict()
                if row_dict in shortlist_rows:
                    continue
                shortlist_rows.append(row_dict)
                if len(shortlist_rows) >= 3:
                    break

    perception_effect_rows = []
    for row in shortlist_rows:
        effect = perception_effect_summary(
            int(row["source_index"]),
            int(row["target_index"]),
            neighbors,
            node_types,
            node_index,
        )
        perception_effect_rows.append({**row, **effect})

    report = {
        "platform_id": PLATFORM_ID,
        "output_subdir": OUTPUT_SUBDIR,
        "gnn_dir": str(gnn_dir),
        "model": "GCN-Encoder + dot-product decoder",
        "epochs_ran": len(history),
        "hidden_dim": args.hidden_dim,
        "confidence_threshold": float(args.confidence_threshold),
        "node_count": int(x.shape[0]),
        "edge_count_after_filter": int(edge_index.shape[1]),
        "train_positive_edges": int(split["train_pos"].shape[1]),
        "validation_positive_edges": int(split["val_pos"].shape[1]),
        "test_positive_edges": int(split["test_pos"].shape[1]),
        "recommendation_count": int(len(shortlist_rows)),
        "perception_effect_count": int(len(perception_effect_rows)),
        "best_validation_auc": float(best_val_auc),
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        },
        "history_tail": history[-5:],
        "source_summary": summary,
    }

    gnn_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(gnn_dir / "gnn_link_training_history.csv", index=False)
    pd.DataFrame(candidate_scores).to_csv(gnn_dir / "gnn_link_predictions.csv", index=False)
    pd.DataFrame(shortlist_rows).head(3).to_csv(gnn_dir / "gnn_link_recommendations.csv", index=False)
    pd.DataFrame(perception_effect_rows).head(3).to_csv(gnn_dir / "gnn_perception_effects.csv", index=False)
    (gnn_dir / "gnn_link_prediction_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "hidden_dim": args.hidden_dim, "feature_dim": int(x.shape[1])}, gnn_dir / "gnn_link_predictor.pt")

    print(f"Trained link predictor on {gnn_dir}")
    print(f"Train AUC: {train_metrics['auc']:.3f} | Val AUC: {val_metrics['auc']:.3f} | Test AUC: {test_metrics['auc']:.3f}")
    print(f"Best validation AUC: {best_val_auc:.3f}")


if __name__ == "__main__":
    main()
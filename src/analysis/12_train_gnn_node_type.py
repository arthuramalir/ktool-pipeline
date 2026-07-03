from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173_synthetic")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
DEFAULT_GNN_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR / "analysis" / "gnn"


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
        from torch_geometric.data import Data
        from torch_geometric.nn import RGCNConv
        return torch, nn, F, Data, RGCNConv
    except ImportError as exc:  # pragma: no cover - graceful runtime guard
        raise SystemExit(
            "PyTorch Geometric training requires `torch` and `torch_geometric`. "
            "Install them first, then rerun this script."
        ) from exc


def load_artifacts(gnn_dir: Path) -> dict[str, Any]:
    summary = load_json(gnn_dir / "gnn_summary.json")
    required = {
        "node_features": gnn_dir / "node_features.npy",
        "edge_index": gnn_dir / "edge_index.npy",
        "edge_type": gnn_dir / "edge_type.npy",
        "edge_weight": gnn_dir / "edge_weight.npy",
        "node_index": gnn_dir / "node_index.csv",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing GNN artifacts: {', '.join(missing)} in {gnn_dir}")

    return {
        "summary": summary,
        "x": np.load(required["node_features"]),
        "edge_index": np.load(required["edge_index"]),
        "edge_type": np.load(required["edge_type"]),
        "edge_weight": np.load(required["edge_weight"]),
        "node_index": pd.read_csv(required["node_index"]),
    }


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


def encode_labels(node_index: pd.DataFrame, exclude_anonymized: bool = True) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    frame = node_index.copy()
    if "node_type" not in frame.columns:
        raise ValueError("node_index.csv must contain a node_type column")
    if exclude_anonymized and "is_anonymized" in frame.columns:
        allowed_mask = ~frame["is_anonymized"].fillna(False).astype(bool)
    else:
        allowed_mask = np.ones(len(frame), dtype=bool)

    labels = frame["node_type"].fillna("unknown").astype(str).to_numpy()
    label_names = sorted(pd.Series(labels[allowed_mask]).unique().tolist())
    label_map = {label: index for index, label in enumerate(label_names)}
    encoded = np.full(len(frame), -1, dtype=np.int64)
    for index, label in enumerate(labels):
        if allowed_mask[index]:
            encoded[index] = label_map[label]
    return encoded, label_map, allowed_mask


def split_masks(labels: np.ndarray, seed: int, train_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(len(labels), dtype=bool)
    val_mask = np.zeros(len(labels), dtype=bool)
    test_mask = np.zeros(len(labels), dtype=bool)

    for label in sorted({int(value) for value in labels.tolist() if value >= 0}):
        class_indices = np.where(labels == label)[0]
        if len(class_indices) == 0:
            continue
        shuffled = class_indices.copy()
        rng.shuffle(shuffled)
        if len(shuffled) < 3:
            train_mask[shuffled] = True
            continue
        train_count = max(1, int(round(len(shuffled) * train_ratio)))
        val_count = max(1, int(round(len(shuffled) * val_ratio)))
        if train_count + val_count >= len(shuffled):
            val_count = max(1, min(val_count, len(shuffled) - train_count - 1))
        train_end = train_count
        val_end = min(len(shuffled), train_end + val_count)
        train_mask[shuffled[:train_end]] = True
        val_mask[shuffled[train_end:val_end]] = True
        test_mask[shuffled[val_end:]] = True

    unused = ~(train_mask | val_mask | test_mask) & (labels >= 0)
    if np.any(unused):
        test_mask[unused] = True
    return train_mask, val_mask, test_mask


def node_confidence(edge_index: np.ndarray, edge_weight: np.ndarray, n_nodes: int) -> np.ndarray:
    if edge_index.size == 0 or edge_weight.size == 0:
        return np.full(n_nodes, 0.5, dtype=np.float32)
    source = edge_index[0].astype(np.int64)
    target = edge_index[1].astype(np.int64)
    weights = edge_weight.astype(np.float32)
    weighted_sum = np.bincount(source, weights=weights, minlength=n_nodes) + np.bincount(target, weights=weights, minlength=n_nodes)
    counts = np.bincount(source, minlength=n_nodes) + np.bincount(target, minlength=n_nodes)
    confidence = np.divide(weighted_sum, counts, out=np.full(n_nodes, 0.5, dtype=np.float32), where=counts > 0)
    return np.clip(confidence.astype(np.float32), 0.0, 1.0)


def weighted_cross_entropy(logits, labels, sample_weights, mask, F):
    active = mask & (labels >= 0)
    if not bool(active.any()):
        return logits.sum() * 0.0
    per_node = F.cross_entropy(logits[active], labels[active], reduction="none")
    weights = sample_weights[active]
    weights = weights / max(float(weights.mean().item()), 1.0e-6)
    return (per_node * weights).mean()


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> dict[str, float]:
    active = y_true >= 0
    y_true = y_true[active]
    y_pred = y_pred[active]
    if y_true.size == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    accuracy = float((y_true == y_pred).mean())
    f1_scores = []
    for label in range(class_count):
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def make_model(nn, RGCNConv, input_dim: int, hidden_dim: int, relation_count: int, class_count: int):
    class NodeTypeRGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = RGCNConv(input_dim, hidden_dim, relation_count, num_bases=max(2, min(8, relation_count)))
            self.conv2 = RGCNConv(hidden_dim, hidden_dim, relation_count, num_bases=max(2, min(8, relation_count)))
            self.dropout = nn.Dropout(0.25)
            self.classifier = nn.Linear(hidden_dim, class_count)

        def forward(self, x, edge_index, edge_type):
            x = self.conv1(x, edge_index, edge_type)
            x = nn.functional.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index, edge_type)
            x = nn.functional.relu(x)
            x = self.dropout(x)
            return self.classifier(x)

    return NodeTypeRGCN()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small R-GCN node classifier on the prepared ALC K-Tool GNN tensors")
    parser.add_argument("--gnn-dir", type=Path, default=DEFAULT_GNN_DIR)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-anonymized", action="store_true", default=True)
    args = parser.parse_args()

    torch, nn, F, Data, RGCNConv = require_torch_packages()
    gnn_dir = discover_gnn_dir(args.gnn_dir)
    artifacts = load_artifacts(gnn_dir)
    summary = artifacts["summary"]

    x = torch.tensor(artifacts["x"], dtype=torch.float32)
    edge_index = torch.tensor(artifacts["edge_index"], dtype=torch.long)
    edge_type = torch.tensor(artifacts["edge_type"], dtype=torch.long)
    edge_weight = torch.tensor(artifacts["edge_weight"], dtype=torch.float32)
    node_index = artifacts["node_index"].copy()

    labels, label_map, label_allowed_mask = encode_labels(node_index, exclude_anonymized=args.exclude_anonymized)
    class_count = len(label_map)
    if class_count < 2:
        raise ValueError("Need at least two node-type classes for classification training.")

    confidence_mask = edge_weight >= float(args.confidence_threshold)
    if confidence_mask.any():
        edge_index = edge_index[:, confidence_mask]
        edge_type = edge_type[confidence_mask]
        edge_weight = edge_weight[confidence_mask]

    if edge_index.numel() == 0:
        raise ValueError("No edges remain after confidence filtering.")

    node_conf = torch.tensor(node_confidence(edge_index.cpu().numpy(), edge_weight.cpu().numpy(), x.shape[0]), dtype=torch.float32)
    train_mask_np, val_mask_np, test_mask_np = split_masks(labels, args.seed, args.train_ratio, args.val_ratio)
    train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
    val_mask = torch.tensor(val_mask_np, dtype=torch.bool)
    test_mask = torch.tensor(test_mask_np, dtype=torch.bool)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    model = make_model(nn, RGCNConv, x.shape[1], args.hidden_dim, int(edge_type.max().item()) + 1, class_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_accuracy = -1.0
    patience = 20
    patience_left = patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index, edge_type)
        loss = weighted_cross_entropy(logits, label_tensor, node_conf, train_mask, F)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits = model(x, edge_index, edge_type)
            predictions = eval_logits.argmax(dim=1)
            train_metrics = classification_metrics(label_tensor[train_mask].cpu().numpy(), predictions[train_mask].cpu().numpy(), class_count)
            val_metrics = classification_metrics(label_tensor[val_mask].cpu().numpy(), predictions[val_mask].cpu().numpy(), class_count)
            test_metrics = classification_metrics(label_tensor[test_mask].cpu().numpy(), predictions[test_mask].cpu().numpy(), class_count)

        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_macro_f1": test_metrics["macro_f1"],
            }
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
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
        logits = model(x, edge_index, edge_type)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

    split_names = np.array(["unassigned"] * len(labels), dtype=object)
    split_names[train_mask_np] = "train"
    split_names[val_mask_np] = "val"
    split_names[test_mask_np] = "test"

    prediction_rows = []
    node_types = node_index["node_type"].fillna("unknown").astype(str).tolist()
    public_labels = node_index["public_label"].fillna(node_index["global_id"]).astype(str).tolist()
    for idx, row in node_index.iterrows():
        pred_index = int(predictions[idx].item()) if labels[idx] >= 0 else -1
        pred_label = next((name for name, index in label_map.items() if index == pred_index), "unassigned") if pred_index >= 0 else "unassigned"
        confidence = float(probabilities[idx].max().item())
        prediction_rows.append(
            {
                "node_index": int(idx),
                "global_id": str(row["global_id"]),
                "public_label": public_labels[idx],
                "node_type": node_types[idx],
                "true_label": node_types[idx] if labels[idx] >= 0 else "unlabeled",
                "predicted_label": pred_label,
                "predicted_class_index": pred_index,
                "split": split_names[idx],
                "prediction_confidence": confidence,
                "node_confidence": float(node_conf[idx].item()),
                "is_anonymized": bool(row.get("is_anonymized", False)),
            }
        )

    final_train = classification_metrics(label_tensor[train_mask].cpu().numpy(), predictions[train_mask].cpu().numpy(), class_count)
    final_val = classification_metrics(label_tensor[val_mask].cpu().numpy(), predictions[val_mask].cpu().numpy(), class_count)
    final_test = classification_metrics(label_tensor[test_mask].cpu().numpy(), predictions[test_mask].cpu().numpy(), class_count)

    report = {
        "platform_id": PLATFORM_ID,
        "output_subdir": OUTPUT_SUBDIR,
        "gnn_dir": str(args.gnn_dir),
        "model": "RGCN",
        "epochs_ran": len(history),
        "hidden_dim": args.hidden_dim,
        "relation_count": int(edge_type.max().item()) + 1,
        "feature_dim": int(x.shape[1]),
        "class_count": class_count,
        "label_map": label_map,
        "confidence_threshold": float(args.confidence_threshold),
        "confidence_filter_edge_count": int(edge_index.shape[1]),
        "node_count": int(x.shape[0]),
        "node_confidence_mean": float(node_conf.mean().item()),
        "metrics": {
            "train": final_train,
            "validation": final_val,
            "test": final_test,
        },
        "best_val_accuracy": best_val_accuracy,
        "history_tail": history[-5:],
        "source_summary": summary,
    }

    gnn_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(gnn_dir / "gnn_node_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(gnn_dir / "gnn_training_history.csv", index=False)
    (gnn_dir / "gnn_training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "label_map": label_map, "feature_dim": int(x.shape[1])}, gnn_dir / "gnn_rgcn_model.pt")

    print(f"Trained R-GCN on {gnn_dir}")
    print(f"Train accuracy: {final_train['accuracy']:.3f} | Val accuracy: {final_val['accuracy']:.3f} | Test accuracy: {final_test['accuracy']:.3f}")
    print(f"Best validation accuracy: {best_val_accuracy:.3f}")


if __name__ == "__main__":
    main()

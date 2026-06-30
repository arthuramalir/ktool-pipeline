from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from strapi_utils import data_dir


class PairDataset(Dataset):
    def __init__(self, features: np.ndarray, pairs: list[tuple[int, int, float]]):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        i, j, label = self.pairs[index]
        return self.features[i], self.features[j], torch.tensor(label, dtype=torch.float32)


class ProjectionModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_embeddings(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    frame = pd.read_parquet(path)
    return np.asarray(frame["embedding"].tolist(), dtype=np.float32)


def build_pairs(documents: pd.DataFrame, graph: nx.Graph, semantic_embeddings: np.ndarray, top_k: int = 5) -> list[tuple[int, int, float]]:
    doc_ids = documents["id"].astype(str).tolist()
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    semantic = StandardScaler().fit_transform(semantic_embeddings.astype(np.float32))
    similarity = cosine_similarity(semantic)
    pair_labels: dict[tuple[int, int], float] = {}

    def add_pair(left: int, right: int, label: float) -> None:
        pair = tuple(sorted((int(left), int(right))))
        current = pair_labels.get(pair)
        if current is None or label > current:
            pair_labels[pair] = label

    for node in graph.nodes():
        if node not in id_to_idx:
            continue
        idx = id_to_idx[node]
        if node in graph:
            for neighbor in graph.neighbors(node):
                if neighbor in id_to_idx:
                    add_pair(idx, id_to_idx[neighbor], 1.0)
        semantic_neighbors = np.argsort(similarity[idx])[::-1][1 : top_k + 1]
        for neighbor_idx in semantic_neighbors:
            add_pair(idx, int(neighbor_idx), 1.0)
        negative_neighbors = np.argsort(similarity[idx])[:top_k]
        for neighbor_idx in negative_neighbors:
            if int(neighbor_idx) != idx:
                add_pair(idx, int(neighbor_idx), -1.0)
    return [(left, right, label) for (left, right), label in pair_labels.items()]


def train_model(features: np.ndarray, pairs: list[tuple[int, int, float]], epochs: int = 10) -> ProjectionModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairDataset(features, pairs)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = ProjectionModel(features.shape[1], output_dim=min(64, max(16, features.shape[1] // 2))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CosineEmbeddingLoss(margin=0.2)
    model.train()
    for _ in range(epochs):
        for left, right, label in loader:
            left = left.to(device)
            right = right.to(device)
            target = label.to(device)
            optimizer.zero_grad()
            left_proj = model(left)
            right_proj = model(right)
            loss = loss_fn(left_proj, right_proj, target)
            loss.backward()
            optimizer.step()
    return model.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a contrastive graph-text projection baseline")
    parser.add_argument("--documents", type=Path, default=data_dir() / "documents.parquet")
    parser.add_argument("--graph", type=Path, default=data_dir() / "graph.graphml")
    parser.add_argument("--semantic-npy", type=Path, default=data_dir() / "embeddings_sentence_transformer.npy")
    parser.add_argument("--semantic-parquet", type=Path, default=data_dir() / "document_embeddings.parquet")
    parser.add_argument("--output", type=Path, default=data_dir() / "contrastive_embeddings.parquet")
    args = parser.parse_args()

    documents = pd.read_parquet(args.documents)
    graph = nx.read_graphml(args.graph).to_undirected() if args.graph.exists() else nx.Graph()
    semantic = load_embeddings(args.semantic_npy if args.semantic_npy.exists() else args.semantic_parquet)
    features = StandardScaler().fit_transform(semantic.astype(np.float32))
    pairs = build_pairs(documents, graph, semantic)
    if not pairs:
        pairs = [(i, i, 1.0) for i in range(len(documents))]

    model = train_model(features, pairs)
    with torch.no_grad():
        embeddings = model(torch.tensor(features, dtype=torch.float32)).numpy()

    frame = pd.DataFrame(
        {
            "entity_id": documents["id"].astype(str).tolist(),
            "contrastive_embedding": [json.dumps([float(x) for x in row.tolist()], ensure_ascii=True) for row in embeddings],
            "embedding_dim": [int(embeddings.shape[1])] * len(embeddings),
        }
    )
    frame.to_parquet(args.output, index=False)
    print(f"Wrote contrastive embeddings to {args.output}")


if __name__ == "__main__":
    main()

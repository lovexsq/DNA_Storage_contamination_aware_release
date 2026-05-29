from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .decode import beam_search_decode, count_scores, edit_distance, fused_scores, greedy_decode
from .dna_data import GraphData
from .models import EdgePredictor, FocalLoss


@dataclass
class TrainSettings:
    backbone: str = "gat"
    hidden_dim: int = 128
    layers: int = 6
    heads: int = 4
    epochs: int = 18
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cpu"
    monitor: str = "srr"
    read_aux_weight: float = 0.0
    read_feature_dim: int = 5


def graph_to_tensors(graph: GraphData, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_features = torch.tensor(graph.node_features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=device)
    edge_features = torch.tensor(graph.edge_features, dtype=torch.float32, device=device)
    labels = torch.tensor(graph.edge_labels, dtype=torch.float32, device=device)
    return node_features, edge_index, edge_features, labels


def graph_to_read_tensors(graph: GraphData, device: str) -> Tuple[torch.Tensor, torch.Tensor] | None:
    if graph.read_features.size == 0 or graph.read_labels.size == 0:
        return None
    if graph.read_features.shape[0] != graph.read_labels.shape[0]:
        return None
    features = torch.tensor(graph.read_features, dtype=torch.float32, device=device)
    labels = torch.tensor(graph.read_labels, dtype=torch.float32, device=device)
    return features, labels


def split_graphs(graphs: Sequence[GraphData], seed: int, train_ratio: float = 0.65, val_ratio: float = 0.15):
    rng = random.Random(seed)
    grouped: Dict[int, List[GraphData]] = {}
    for graph in graphs:
        grouped.setdefault(graph.cluster_id, []).append(graph)
    cluster_ids = list(grouped)
    rng.shuffle(cluster_ids)
    n_train = int(len(cluster_ids) * train_ratio)
    n_val = int(len(cluster_ids) * val_ratio)
    train_ids = set(cluster_ids[:n_train])
    val_ids = set(cluster_ids[n_train : n_train + n_val])
    test_ids = set(cluster_ids[n_train + n_val :])
    train = [graph for graph in graphs if graph.cluster_id in train_ids]
    val = [graph for graph in graphs if graph.cluster_id in val_ids]
    test = [graph for graph in graphs if graph.cluster_id in test_ids]
    return train, val, test


def edge_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float]:
    preds = torch.sigmoid(logits) >= 0.5
    truth = labels >= 0.5
    tp = float((preds & truth).sum().item())
    fp = float((preds & ~truth).sum().item())
    fn = float((~preds & truth).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


@torch.no_grad()
def evaluate_edges(model: EdgePredictor, graphs: Sequence[GraphData], device: str) -> Dict[str, float]:
    model.eval()
    losses = []
    precisions = []
    recalls = []
    f1s = []
    criterion = FocalLoss()
    for graph in graphs:
        node_features, edge_index, edge_features, labels = graph_to_tensors(graph, device)
        logits = model(node_features, edge_index, edge_features)
        losses.append(float(criterion(logits, labels).item()))
        precision, recall, f1 = edge_f1_from_logits(logits, labels)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "edge_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


@torch.no_grad()
def evaluate_read_contamination(model: EdgePredictor, graphs: Sequence[GraphData], device: str) -> Dict[str, float]:
    if getattr(model, "read_head", None) is None:
        return {
            "read_loss": 0.0,
            "read_accuracy": 0.0,
            "read_precision": 0.0,
            "read_recall": 0.0,
            "read_f1": 0.0,
        }
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    losses = []
    tp = fp = fn = tn = 0.0
    for graph in graphs:
        tensors = graph_to_read_tensors(graph, device)
        if tensors is None:
            continue
        read_features, read_labels = tensors
        logits = model.predict_read_contamination(read_features)
        losses.append(float(criterion(logits, read_labels).item()))
        preds = torch.sigmoid(logits) >= 0.5
        truth = read_labels >= 0.5
        tp += float((preds & truth).sum().item())
        fp += float((preds & ~truth).sum().item())
        fn += float((~preds & truth).sum().item())
        tn += float((~preds & ~truth).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1.0)
    return {
        "read_loss": float(np.mean(losses)) if losses else 0.0,
        "read_accuracy": accuracy,
        "read_precision": precision,
        "read_recall": recall,
        "read_f1": f1,
    }


@torch.no_grad()
def validation_srr(model: EdgePredictor, graphs: Sequence[GraphData], device: str, beam_width: int = 8) -> float:
    model.eval()
    exact = []
    for graph in graphs:
        node_features, edge_index, edge_features, _ = graph_to_tensors(graph, device)
        scores = torch.sigmoid(model(node_features, edge_index, edge_features)).detach().cpu().numpy()
        prediction = beam_search_decode(graph, scores, beam_width=beam_width)
        exact.append(1.0 if prediction == graph.reference else 0.0)
    return float(np.mean(exact)) if exact else 0.0


def train_model(
    train_graphs: Sequence[GraphData],
    val_graphs: Sequence[GraphData],
    settings: TrainSettings,
    checkpoint_path: Path,
    history_path: Path,
    initial_state_path: Path | None = None,
) -> EdgePredictor:
    torch.manual_seed(settings.seed)
    random.seed(settings.seed)
    np.random.seed(settings.seed)

    node_dim = train_graphs[0].node_features.shape[1]
    edge_dim = train_graphs[0].edge_features.shape[1]
    model = EdgePredictor(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=settings.hidden_dim,
        layers=settings.layers,
        backbone=settings.backbone,
        heads=settings.heads,
        use_read_head=settings.read_aux_weight > 0.0,
        read_feature_dim=settings.read_feature_dim,
    ).to(settings.device)
    if initial_state_path is not None:
        model.load_state_dict(torch.load(initial_state_path, map_location=settings.device), strict=False)
    criterion = FocalLoss(alpha=0.70, gamma=2.0)
    read_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0
    history_rows = []

    for epoch in range(1, settings.epochs + 1):
        model.train()
        shuffled = list(train_graphs)
        random.shuffle(shuffled)
        train_losses = []
        train_edge_losses = []
        train_read_losses = []
        for graph in shuffled:
            node_features, edge_index, edge_features, labels = graph_to_tensors(graph, settings.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(node_features, edge_index, edge_features)
            edge_loss = criterion(logits, labels)
            loss = edge_loss
            read_loss_value = 0.0
            if settings.read_aux_weight > 0.0 and getattr(model, "read_head", None) is not None:
                read_tensors = graph_to_read_tensors(graph, settings.device)
                if read_tensors is not None:
                    read_features, read_labels = read_tensors
                    read_logits = model.predict_read_contamination(read_features)
                    positives = read_labels.sum()
                    negatives = read_labels.numel() - positives
                    if float(positives.item()) > 0.0:
                        pos_weight = (negatives / positives.clamp(min=1.0)).detach()
                        read_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            read_logits,
                            read_labels,
                            pos_weight=pos_weight,
                        )
                    else:
                        read_loss = read_criterion(read_logits, read_labels)
                    loss = loss + settings.read_aux_weight * read_loss
                    read_loss_value = float(read_loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_edge_losses.append(float(edge_loss.item()))
            train_read_losses.append(read_loss_value)

        val_metrics = evaluate_edges(model, val_graphs, settings.device)
        val_read_metrics = evaluate_read_contamination(model, val_graphs, settings.device)
        val_srr = validation_srr(model, val_graphs, settings.device, beam_width=8)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_edge_loss": float(np.mean(train_edge_losses)),
            "train_read_loss": float(np.mean(train_read_losses)) if train_read_losses else 0.0,
            "val_loss": val_metrics["loss"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_edge_f1": val_metrics["edge_f1"],
            "val_read_loss": val_read_metrics["read_loss"],
            "val_read_f1": val_read_metrics["read_f1"],
            "val_srr": val_srr,
        }
        history_rows.append(row)
        print(
            f"[{settings.backbone}] 第 {epoch:02d} 轮 "
            f"训练loss={row['train_loss']:.4f} "
            f"验证Edge-F1={row['val_edge_f1']:.4f} "
            f"验证Read-F1={row['val_read_f1']:.4f} "
            f"验证SRR={row['val_srr']:.4f}"
        )
        metric = row["val_srr"] if settings.monitor == "srr" else row["val_edge_f1"]
        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), checkpoint_path)

    with history_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)
    model.load_state_dict(torch.load(checkpoint_path, map_location=settings.device), strict=False)
    return model


@torch.no_grad()
def predict_edge_scores(model: EdgePredictor, graph: GraphData, device: str) -> np.ndarray:
    model.eval()
    node_features, edge_index, edge_features, _ = graph_to_tensors(graph, device)
    logits = model(node_features, edge_index, edge_features)
    return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_edge_scores_mc_dropout(
    model: EdgePredictor,
    graph: GraphData,
    device: str,
    passes: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    was_training = model.training
    model.train()
    node_features, edge_index, edge_features, _ = graph_to_tensors(graph, device)
    predictions = []
    for _ in range(max(int(passes), 1)):
        logits = model(node_features, edge_index, edge_features)
        predictions.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
    if not was_training:
        model.eval()
    stacked = np.stack(predictions, axis=0)
    mean_scores = stacked.mean(axis=0).astype(np.float32)
    variance = stacked.var(axis=0).astype(np.float32)
    uncertainty = np.clip(variance / 0.25, 0.0, 1.0).astype(np.float32)
    return mean_scores, uncertainty


def evaluate_reconstruction(
    model: EdgePredictor,
    graphs: Sequence[GraphData],
    device: str,
    include_learned: bool = True,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for graph in graphs:
        model_scores = predict_edge_scores(model, graph, device) if include_learned else count_scores(graph)
        methods = {
            "Count Greedy": greedy_decode(graph, count_scores(graph)),
            "Count Beam": beam_search_decode(graph, count_scores(graph), beam_width=8),
            "GNN Greedy": greedy_decode(graph, model_scores),
            "GNN Beam": beam_search_decode(graph, model_scores, beam_width=8),
            "GNN+Fixed Beam": beam_search_decode(graph, fused_scores(graph, model_scores, "fixed"), beam_width=8),
            "GNN+Dynamic Beam": beam_search_decode(graph, fused_scores(graph, model_scores, "dynamic"), beam_width=8),
        }
        for method, prediction in methods.items():
            distance = edit_distance(prediction, graph.reference)
            rows.append(
                {
                    "cluster_id": graph.cluster_id,
                    "coverage": graph.coverage,
                    "k": graph.k,
                    "method": method,
                    "reference": graph.reference,
                    "prediction": prediction,
                    "exact": 1.0 if prediction == graph.reference else 0.0,
                    "edit_distance": distance,
                    "base_recovery_rate": 1.0 - distance / max(len(graph.reference), 1),
                    "num_nodes": len(graph.node_strings),
                    "num_edges": graph.edge_index.shape[1],
                    "positive_edge_ratio": float(graph.edge_labels.mean()),
                }
            )
    return rows


def summarize(rows: Sequence[Dict[str, object]], group_keys: Iterable[str]) -> List[Dict[str, object]]:
    group_keys = list(group_keys)
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[item] for item in group_keys)
        grouped.setdefault(key, []).append(row)
    summary = []
    for key, selected in grouped.items():
        out = {name: value for name, value in zip(group_keys, key)}
        out.update(
            {
                "samples": len(selected),
                "sequence_recovery_rate": float(np.mean([float(row["exact"]) for row in selected])),
                "base_recovery_rate": float(np.mean([float(row["base_recovery_rate"]) for row in selected])),
                "avg_edit_distance": float(np.mean([float(row["edit_distance"]) for row in selected])),
                "avg_nodes": float(np.mean([float(row["num_nodes"]) for row in selected])),
                "avg_edges": float(np.mean([float(row["num_edges"]) for row in selected])),
            }
        )
        summary.append(out)
    return sorted(summary, key=lambda row: tuple(str(row[key]) for key in group_keys))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

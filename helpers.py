from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.decode import edit_distance
from src.dna_data import GraphData, ReadCluster


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_URL = "https://github.com/microsoft/clustered-nanopore-reads-dataset"


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def complete_cluster_ids(
    clusters: Sequence[ReadCluster],
    graphs: Sequence[GraphData],
    k_values: Sequence[int],
    seed: int,
) -> List[int]:
    required = set(k_values)
    available: Dict[int, set[int]] = defaultdict(set)
    for graph in graphs:
        available[graph.cluster_id].add(graph.k)
    ids = [cluster.cluster_id for cluster in clusters if required.issubset(available.get(cluster.cluster_id, set()))]
    random.Random(seed).shuffle(ids)
    return ids


def metric_row(
    graph: GraphData,
    method: str,
    prediction: str,
    extra: Dict[str, object] | None = None,
) -> Dict[str, object]:
    distance = edit_distance(prediction, graph.reference)
    row: Dict[str, object] = {
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
    if extra:
        row.update(extra)
    return row


def dataset_profile(graphs: Sequence[GraphData], label: str) -> List[Dict[str, object]]:
    if not graphs:
        return []
    coverages = np.asarray([graph.coverage for graph in graphs], dtype=np.float32)
    nodes = np.asarray([len(graph.node_strings) for graph in graphs], dtype=np.float32)
    edges = np.asarray([graph.edge_index.shape[1] for graph in graphs], dtype=np.float32)
    positives = np.asarray([float(graph.edge_labels.mean()) for graph in graphs], dtype=np.float32)
    return [
        {
            "dataset": label,
            "graphs": len(graphs),
            "unique_clusters": len({graph.cluster_id for graph in graphs}),
            "avg_coverage": float(coverages.mean()),
            "min_coverage": float(coverages.min()),
            "max_coverage": float(coverages.max()),
            "avg_nodes": float(nodes.mean()),
            "avg_edges": float(edges.mean()),
            "avg_positive_edge_ratio": float(positives.mean()),
        }
    ]

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from helpers import DATASET_URL, complete_cluster_ids, dataset_profile, resolve_project_path
from run_contamination_aware_experiment import evaluate_count_graphs, evaluate_model_graphs
from src.dna_data import (
    GraphData,
    ReadCluster,
    contaminate_clusters,
    graphs_from_clusters,
    load_cnr_clusters,
)
from src.models import EdgePredictor
from src.train_eval import evaluate_edges, evaluate_read_contamination, summarize, write_csv


def select_clusters(clusters: Sequence[ReadCluster], selected_ids: Sequence[int]) -> List[ReadCluster]:
    selected = set(selected_ids)
    return [cluster for cluster in clusters if cluster.cluster_id in selected]


def load_mpnn(
    checkpoint_path: Path,
    graph: GraphData,
    args: argparse.Namespace,
    use_read_head: bool,
) -> EdgePredictor:
    model = EdgePredictor(
        node_dim=graph.node_features.shape[1],
        edge_dim=graph.edge_features.shape[1],
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        backbone="mpnn",
        heads=args.heads,
        use_read_head=use_read_head,
        read_feature_dim=5,
    ).to(args.device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device), strict=False)
    model.eval()
    return model


def build_graphs(
    clusters: Sequence[ReadCluster],
    args: argparse.Namespace,
    read_weighting: bool,
    attach_read_features: bool,
) -> List[GraphData]:
    return graphs_from_clusters(
        list(clusters),
        args.k_values,
        threshold=args.threshold,
        indel_aware=True,
        max_indel_shift=args.max_indel_shift,
        read_weighting=read_weighting,
        attach_read_features=attach_read_features,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="评估发布包内置的污染感知 MPNN checkpoint。")
    parser.add_argument("--raw-dir", default="data/raw/clustered-nanopore-reads-dataset")
    parser.add_argument("--train-clusters", type=int, default=1000)
    parser.add_argument("--val-clusters", type=int, default=150)
    parser.add_argument("--test-clusters", type=int, default=300)
    parser.add_argument("--load-buffer", type=int, default=1000)
    parser.add_argument("--real-min-reads", type=int, default=20)
    parser.add_argument("--real-max-reads-per-cluster", type=int, default=20)
    parser.add_argument("--test-contamination-rates", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--contamination-mode", choices=["replace", "append"], default="replace")
    parser.add_argument("--k-values", type=int, nargs="+", default=[11, 13, 15])
    parser.add_argument("--primary-k", type=int, default=13)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--max-indel-shift", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mc-passes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints/contamination_aware_train1000_test300_with_clean")
    parser.add_argument("--results-dir", default="results/evaluate_included_checkpoints")
    args = parser.parse_args()

    started_at = time.time()
    raw_dir = resolve_project_path(args.raw_dir)
    checkpoint_dir = resolve_project_path(args.checkpoint_dir)
    results_dir = resolve_project_path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    requested_clusters = args.train_clusters + args.val_clusters + args.test_clusters
    load_limit = requested_clusters + max(args.load_buffer, 0)
    print(f"正在读取最多 {load_limit} 个 CNR 簇...", flush=True)
    real_clusters = load_cnr_clusters(
        raw_dir=raw_dir,
        max_clusters=load_limit,
        min_reads=args.real_min_reads,
        max_reads_per_cluster=args.real_max_reads_per_cluster,
        seed=args.seed,
    )
    clean_graphs = graphs_from_clusters(
        real_clusters,
        args.k_values,
        threshold=args.threshold,
        indel_aware=True,
        max_indel_shift=args.max_indel_shift,
    )
    usable_ids = complete_cluster_ids(real_clusters, clean_graphs, args.k_values, seed=args.seed)
    if len(usable_ids) < requested_clusters:
        raise ValueError(f"当前只有 {len(usable_ids)} 个完整可用簇。")

    test_start = args.train_clusters + args.val_clusters
    test_ids = usable_ids[test_start : test_start + args.test_clusters]
    clean_test_clusters = select_clusters(real_clusters, test_ids)
    print(f"正在评估 {len(test_ids)} 个测试簇。", flush=True)

    detail_rows = []
    edge_rows = []
    read_rows = []
    profiles = []
    models: Dict[str, EdgePredictor] | None = None

    for rate in args.test_contamination_rates:
        print(f"正在评估污染率 {rate:.2f}...", flush=True)
        contaminated_test_clusters = contaminate_clusters(
            clean_test_clusters,
            contamination_rate=rate,
            seed=args.seed + int(rate * 10000) + 4000,
            mode=args.contamination_mode,
            preserve_ids=True,
        )
        unweighted_test = build_graphs(contaminated_test_clusters, args, read_weighting=False, attach_read_features=False)
        weighted_test = build_graphs(contaminated_test_clusters, args, read_weighting=True, attach_read_features=True)
        profiles.extend(dataset_profile(unweighted_test, f"test_unweighted_contam_{rate:.2f}"))
        profiles.extend(dataset_profile(weighted_test, f"test_weighted_contam_{rate:.2f}"))
        detail_rows.extend(evaluate_count_graphs(unweighted_test, "Unweighted", args.primary_k, rate))
        detail_rows.extend(evaluate_count_graphs(weighted_test, "Read-weighted", args.primary_k, rate))

        if models is None:
            models = {
                "Read-weighted MPNN": load_mpnn(
                    checkpoint_dir / "read_weighted_mpnn_transfer_best.pt",
                    weighted_test[0],
                    args,
                    use_read_head=False,
                ),
                "Read-weighted MPNN+ReadAux": load_mpnn(
                    checkpoint_dir / "read_weighted_mpnn_readaux_transfer_best.pt",
                    weighted_test[0],
                    args,
                    use_read_head=True,
                ),
            }

        for label, model in models.items():
            metrics = evaluate_edges(model, weighted_test, args.device)
            metrics.update({"model": label, "contamination_rate": rate})
            edge_rows.append(metrics)
            read_metrics = evaluate_read_contamination(model, weighted_test, args.device)
            read_metrics.update({"model": label, "contamination_rate": rate})
            read_rows.append(read_metrics)
            detail_rows.extend(
                evaluate_model_graphs(
                    model,
                    weighted_test,
                    label,
                    args.device,
                    mc_passes=args.mc_passes,
                    primary_k=args.primary_k,
                    contamination_rate=rate,
                )
            )

    summary_rows = summarize(detail_rows, ["contamination_rate", "method"])
    write_csv(results_dir / "dataset_profile.csv", profiles)
    write_csv(results_dir / "edge_metrics.csv", edge_rows)
    write_csv(results_dir / "read_contamination_metrics.csv", read_rows)
    write_csv(results_dir / "method_detail.csv", detail_rows)
    write_csv(results_dir / "method_summary.csv", summary_rows)

    elapsed_seconds = time.time() - started_at
    metadata = {
        "dataset": "Microsoft Clustered Nanopore Reads (CNR)",
        "dataset_url": DATASET_URL,
        "args": vars(args),
        "loaded_real_clusters": len(real_clusters),
        "usable_complete_clusters": len(usable_ids),
        "test_clusters": len(test_ids),
        "loaded_from_checkpoint": True,
        "elapsed_seconds": elapsed_seconds,
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("方法汇总:", flush=True)
    for row in summary_rows:
        print(
            f"污染率={float(row['contamination_rate']):.2f} {row['method']}: "
            f"SRR={row['sequence_recovery_rate']:.3f}, "
            f"BRR={row['base_recovery_rate']:.3f}, AvgED={row['avg_edit_distance']:.3f}",
            flush=True,
        )
    print(f"耗时分钟数: {elapsed_seconds / 60:.1f}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    main()

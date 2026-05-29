from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from helpers import DATASET_URL, complete_cluster_ids, dataset_profile, metric_row, resolve_project_path
from src.decode import beam_search_decode, count_scores, greedy_decode, uncertainty_fused_scores
from src.dna_data import (
    GraphData,
    ReadCluster,
    contaminate_clusters,
    graphs_from_clusters,
    load_cnr_clusters,
    make_graph_dataset,
)
from src.train_eval import (
    TrainSettings,
    evaluate_edges,
    evaluate_read_contamination,
    predict_edge_scores,
    predict_edge_scores_mc_dropout,
    split_graphs,
    summarize,
    train_model,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def select_clusters(clusters: Sequence[ReadCluster], selected_ids: Sequence[int]) -> List[ReadCluster]:
    order = {cluster_id: index for index, cluster_id in enumerate(selected_ids)}
    selected = [cluster for cluster in clusters if cluster.cluster_id in order]
    return sorted(selected, key=lambda cluster: order[cluster.cluster_id])


def build_augmented_clusters(
    clusters: Sequence[ReadCluster],
    rates: Sequence[float],
    seed: int,
    mode: str,
    cluster_id_offset: int,
) -> List[ReadCluster]:
    augmented: List[ReadCluster] = []
    stride = max(len(clusters), 1)
    for rate_index, rate in enumerate(rates):
        augmented.extend(
            contaminate_clusters(
                clusters,
                contamination_rate=rate,
                seed=seed + rate_index * 997,
                mode=mode,
                preserve_ids=False,
                cluster_id_offset=cluster_id_offset + rate_index * stride,
            )
        )
    return augmented


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
        indel_aware=args.indel_aware,
        max_indel_shift=args.max_indel_shift,
        read_weighting=read_weighting,
        attach_read_features=attach_read_features,
    )


def primary_graphs(graphs: Sequence[GraphData], primary_k: int) -> List[GraphData]:
    grouped: Dict[int, List[GraphData]] = defaultdict(list)
    for graph in graphs:
        grouped[graph.cluster_id].append(graph)
    selected = []
    for cluster_id in sorted(grouped):
        bundle = grouped[cluster_id]
        selected.append(min(bundle, key=lambda graph: (abs(graph.k - primary_k), graph.k)))
    return selected


def evaluate_count_graphs(
    graphs: Sequence[GraphData],
    method_prefix: str,
    primary_k: int,
    contamination_rate: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for graph in primary_graphs(graphs, primary_k):
        scores = count_scores(graph)
        extra = {
            "contamination_rate": contamination_rate,
            "weighted_graph": 1.0 if graph.weighted_graph else 0.0,
            "graph_contamination_rate": graph.contamination_rate,
            "avg_read_weight": float(np.mean(graph.read_weights)) if graph.read_weights.size else 1.0,
        }
        rows.append(metric_row(graph, f"{method_prefix} Count Greedy", greedy_decode(graph, scores), extra))
        rows.append(metric_row(graph, f"{method_prefix} Count Beam", beam_search_decode(graph, scores, beam_width=8), extra))
    return rows


def evaluate_model_graphs(
    model,
    graphs: Sequence[GraphData],
    label: str,
    device: str,
    mc_passes: int,
    primary_k: int,
    contamination_rate: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for graph in primary_graphs(graphs, primary_k):
        model_scores = predict_edge_scores(model, graph, device)
        mean_scores, uncertainty = predict_edge_scores_mc_dropout(model, graph, device, passes=mc_passes)
        uncertainty_scores = uncertainty_fused_scores(graph, mean_scores, uncertainty)
        extra = {
            "contamination_rate": contamination_rate,
            "weighted_graph": 1.0 if graph.weighted_graph else 0.0,
            "graph_contamination_rate": graph.contamination_rate,
            "avg_uncertainty": float(np.mean(uncertainty)),
            "max_uncertainty": float(np.max(uncertainty)),
            "avg_read_weight": float(np.mean(graph.read_weights)) if graph.read_weights.size else 1.0,
        }
        rows.append(metric_row(graph, f"{label} Beam", beam_search_decode(graph, model_scores, beam_width=8), extra))
        rows.append(
            metric_row(
                graph,
                f"{label} Uncertainty Beam",
                beam_search_decode(graph, uncertainty_scores, beam_width=8),
                extra,
            )
        )
    return rows


def train_weighted_mpnn(
    name: str,
    sim_train: Sequence[GraphData],
    sim_val: Sequence[GraphData],
    real_train: Sequence[GraphData],
    real_val: Sequence[GraphData],
    args: argparse.Namespace,
    checkpoints_dir: Path,
    results_dir: Path,
    read_aux_weight: float,
):
    base_settings = dict(
        backbone="mpnn",
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        monitor="srr",
        read_aux_weight=read_aux_weight,
        read_feature_dim=5,
    )
    pretrain_settings = TrainSettings(
        **base_settings,
        epochs=args.pretrain_epochs,
        lr=args.lr,
    )
    pretrain_path = checkpoints_dir / f"{name}_sim_pretrain_best.pt"
    train_model(
        train_graphs=sim_train,
        val_graphs=sim_val,
        settings=pretrain_settings,
        checkpoint_path=pretrain_path,
        history_path=results_dir / f"{name}_sim_pretrain_history.csv",
    )

    finetune_settings = TrainSettings(
        **base_settings,
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
    )
    return train_model(
        train_graphs=real_train,
        val_graphs=real_val,
        settings=finetune_settings,
        checkpoint_path=checkpoints_dir / f"{name}_transfer_best.pt",
        history_path=results_dir / f"{name}_transfer_history.csv",
        initial_state_path=pretrain_path,
    )


def contamination_profile(graphs: Sequence[GraphData], label: str) -> List[Dict[str, object]]:
    if not graphs:
        return []
    rates = np.asarray([graph.contamination_rate for graph in graphs], dtype=np.float32)
    weights = np.asarray(
        [float(np.mean(graph.read_weights)) if graph.read_weights.size else 1.0 for graph in graphs],
        dtype=np.float32,
    )
    return [
        {
            "dataset": label,
            "graphs": len(graphs),
            "unique_clusters": len({graph.cluster_id for graph in graphs}),
            "avg_contamination_rate": float(rates.mean()),
            "min_contamination_rate": float(rates.min()),
            "max_contamination_rate": float(rates.max()),
            "avg_read_weight": float(weights.mean()),
            "min_avg_read_weight": float(weights.min()),
            "max_avg_read_weight": float(weights.max()),
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="运行污染感知 read-weighted graph 序列重构实验。")
    parser.add_argument("--raw-dir", default="data/raw/clustered-nanopore-reads-dataset")
    parser.add_argument("--train-clusters", type=int, default=100)
    parser.add_argument("--val-clusters", type=int, default=20)
    parser.add_argument("--test-clusters", type=int, default=50)
    parser.add_argument("--load-buffer", type=int, default=300)
    parser.add_argument("--real-min-reads", type=int, default=20)
    parser.add_argument("--real-max-reads-per-cluster", type=int, default=20)
    parser.add_argument("--sim-clusters-per-coverage", type=int, default=24)
    parser.add_argument("--sim-coverages", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--sim-contamination-rate", type=float, default=0.20)
    parser.add_argument("--train-contamination-rates", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--val-contamination-rate", type=float, default=0.20)
    parser.add_argument("--test-contamination-rates", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--contamination-mode", choices=["replace", "append"], default="replace")
    parser.add_argument("--k-values", type=int, nargs="+", default=[11, 13, 15])
    parser.add_argument("--primary-k", type=int, default=13)
    parser.add_argument("--threshold", type=int, default=2)
    parser.set_defaults(indel_aware=True)
    parser.add_argument("--no-indel-aware", action="store_false", dest="indel_aware")
    parser.add_argument("--max-indel-shift", type=int, default=3)
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mc-passes", type=int, default=8)
    parser.add_argument("--aux-weight", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results-dir", default="results/contamination_aware_train100_test50")
    parser.add_argument("--checkpoint-dir", default="checkpoints/contamination_aware_train100_test50")
    parser.add_argument("--skip-no-aux-model", action="store_true")
    parser.add_argument("--skip-aux-model", action="store_true")
    args = parser.parse_args()

    started_at = time.time()
    raw_dir = resolve_project_path(args.raw_dir)
    results_dir = resolve_project_path(args.results_dir)
    checkpoints_dir = resolve_project_path(args.checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print("正在构造带污染的模拟预训练图...", flush=True)
    sim_graphs = make_graph_dataset(
        num_clusters=args.sim_clusters_per_coverage,
        coverage_values=args.sim_coverages,
        k_values=args.k_values,
        seed=args.seed + 1000,
        sequence_length=110,
        substitution_rate=0.025,
        insertion_rate=0.020,
        deletion_rate=0.025,
        contaminant_rate=args.sim_contamination_rate,
        indel_aware=args.indel_aware,
        max_indel_shift=args.max_indel_shift,
        read_weighting=True,
        attach_read_features=True,
    )
    sim_train, sim_val, _ = split_graphs(sim_graphs, seed=args.seed + 10)

    requested_clusters = args.train_clusters + args.val_clusters + args.test_clusters
    load_limit = requested_clusters + max(args.load_buffer, 0)
    print(f"正在读取最多 {load_limit} 个真实 CNR 簇...", flush=True)
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
        indel_aware=args.indel_aware,
        max_indel_shift=args.max_indel_shift,
    )
    usable_ids = complete_cluster_ids(real_clusters, clean_graphs, args.k_values, seed=args.seed)
    if len(usable_ids) < requested_clusters:
        raise ValueError(
            f"当前只有 {len(usable_ids)} 个簇能生成所有指定 k 的图，实际需要 {requested_clusters} 个簇。"
        )

    train_ids = usable_ids[: args.train_clusters]
    val_start = args.train_clusters
    val_end = val_start + args.val_clusters
    val_ids = usable_ids[val_start:val_end]
    test_ids = usable_ids[val_end : val_end + args.test_clusters]

    clean_train_clusters = select_clusters(real_clusters, train_ids)
    clean_val_clusters = select_clusters(real_clusters, val_ids)
    clean_test_clusters = select_clusters(real_clusters, test_ids)

    augmented_train_clusters = build_augmented_clusters(
        clean_train_clusters,
        args.train_contamination_rates,
        seed=args.seed + 2000,
        mode=args.contamination_mode,
        cluster_id_offset=1_000_000,
    )
    contaminated_val_clusters = contaminate_clusters(
        clean_val_clusters,
        contamination_rate=args.val_contamination_rate,
        seed=args.seed + 3000,
        mode=args.contamination_mode,
        preserve_ids=True,
    )
    real_train = build_graphs(augmented_train_clusters, args, read_weighting=True, attach_read_features=True)
    real_val = build_graphs(contaminated_val_clusters, args, read_weighting=True, attach_read_features=True)
    if not sim_train or not sim_val or not real_train or not real_val:
        raise ValueError("可用图数量不足。请增加数据读取上限，或降低过滤阈值。")

    print(
        f"模拟图={len(sim_graphs)} 训练图={len(real_train)} "
        f"验证图={len(real_val)} 测试簇={len(clean_test_clusters)}",
        flush=True,
    )

    models = {}
    if not args.skip_no_aux_model:
        print("正在训练 read-weighted Edge-aware MPNN（不使用辅助 loss）...", flush=True)
        models["Read-weighted MPNN"] = train_weighted_mpnn(
            "read_weighted_mpnn",
            sim_train,
            sim_val,
            real_train,
            real_val,
            args,
            checkpoints_dir,
            results_dir,
            read_aux_weight=0.0,
        )
    if not args.skip_aux_model:
        print("正在训练 read-weighted Edge-aware MPNN（使用 read auxiliary loss）...", flush=True)
        models["Read-weighted MPNN+ReadAux"] = train_weighted_mpnn(
            "read_weighted_mpnn_readaux",
            sim_train,
            sim_val,
            real_train,
            real_val,
            args,
            checkpoints_dir,
            results_dir,
            read_aux_weight=args.aux_weight,
        )

    detail_rows: List[Dict[str, object]] = []
    edge_rows: List[Dict[str, object]] = []
    read_rows: List[Dict[str, object]] = []
    profiles: List[Dict[str, object]] = []
    profiles.extend(dataset_profile(sim_graphs, "simulated_contaminated_pretraining"))
    profiles.extend(contamination_profile(sim_graphs, "simulated_contaminated_pretraining"))
    profiles.extend(dataset_profile(real_train, "real_contaminated_train"))
    profiles.extend(contamination_profile(real_train, "real_contaminated_train"))
    profiles.extend(dataset_profile(real_val, "real_contaminated_validation"))
    profiles.extend(contamination_profile(real_val, "real_contaminated_validation"))

    for rate in args.test_contamination_rates:
        print(f"正在评估污染率 {rate:.2f}...", flush=True)
        contaminated_test_clusters = contaminate_clusters(
            clean_test_clusters,
            contamination_rate=rate,
            seed=args.seed + int(rate * 10000) + 4000,
            mode=args.contamination_mode,
            preserve_ids=True,
        )
        unweighted_test = build_graphs(
            contaminated_test_clusters,
            args,
            read_weighting=False,
            attach_read_features=False,
        )
        weighted_test = build_graphs(
            contaminated_test_clusters,
            args,
            read_weighting=True,
            attach_read_features=True,
        )
        detail_rows.extend(evaluate_count_graphs(unweighted_test, "Unweighted", args.primary_k, rate))
        detail_rows.extend(evaluate_count_graphs(weighted_test, "Read-weighted", args.primary_k, rate))
        profiles.extend(dataset_profile(unweighted_test, f"test_unweighted_contam_{rate:.2f}"))
        profiles.extend(dataset_profile(weighted_test, f"test_weighted_contam_{rate:.2f}"))
        profiles.extend(contamination_profile(weighted_test, f"test_weighted_contam_{rate:.2f}"))

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
        "train_clusters": len(train_ids),
        "val_clusters": len(val_ids),
        "test_clusters": len(test_ids),
        "train_graphs": len(real_train),
        "val_graphs": len(real_val),
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
    main()

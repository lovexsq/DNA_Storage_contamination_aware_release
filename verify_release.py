from __future__ import annotations

from pathlib import Path


REQUIRED_PATHS = [
    "data/raw/clustered-nanopore-reads-dataset/Centers.txt",
    "data/raw/clustered-nanopore-reads-dataset/Clusters.txt",
    "checkpoints/contamination_aware_train1000_test300_with_clean/read_weighted_mpnn_transfer_best.pt",
    "checkpoints/contamination_aware_train1000_test300_with_clean/read_weighted_mpnn_readaux_transfer_best.pt",
    "checkpoints/contamination_aware_train100_test50_with_clean/read_weighted_mpnn_transfer_best.pt",
    "checkpoints/contamination_aware_train100_test50_with_clean/read_weighted_mpnn_readaux_transfer_best.pt",
    "results/contamination_aware_train1000_test300_with_clean/method_summary.csv",
    "results/contamination_aware_train1000_test300_with_clean/read_contamination_metrics.csv",
    "src/dna_data.py",
    "src/models.py",
    "src/train_eval.py",
    "src/decode.py",
    "run_contamination_aware_experiment.py",
    "evaluate_existing_checkpoints.py",
    "generate_enhanced_analysis.py",
    "results/enhanced_analysis/analysis_report.md",
    "results/enhanced_analysis/tables/main_metrics_srr.csv",
    "results/enhanced_analysis/tables/main_metrics_brr.csv",
    "results/enhanced_analysis/tables/main_metrics_avged.csv",
    "results/enhanced_analysis/tables/read_weighted_graph_ablation.csv",
    "results/enhanced_analysis/tables/edge_aware_mpnn_contribution.csv",
    "results/enhanced_analysis/tables/read_aux_analysis.csv",
    "results/enhanced_analysis/tables/train_size_100_clusters.csv",
    "results/enhanced_analysis/tables/train_size_1000_clusters.csv",
    "results/enhanced_analysis/tables/case_studies.csv",
    "results/train100_model_on_same300test_as_train1000/method_summary.csv",
    "results/train1000_model_reeval_same300test/method_summary.csv",
    "results/training_scale_same300_comparison.csv",
    "results/enhanced_analysis/figures/srr_by_contamination.png",
    "results/enhanced_analysis/figures/avged_by_contamination.png",
    "results/enhanced_analysis/figures/edge_f1_by_contamination.png",
    "results/enhanced_analysis/figures/read_f1_by_contamination.png",
    "results/enhanced_analysis/figures/avg_edges_by_contamination.png",
    "results/enhanced_analysis/figures/train_size_read_weighted_count_beam.png",
    "results/enhanced_analysis/figures/train_size_read_weighted_mpnn_beam.png",
    "results/enhanced_analysis/figures/train_size_mpnn_readaux_beam.png",
]


def main() -> None:
    root = Path(__file__).resolve().parent
    missing = [path for path in REQUIRED_PATHS if not (root / path).exists()]
    if missing:
        for path in missing:
            print(f"缺少文件: {path}")
        raise SystemExit(1)

    import numpy
    import torch

    print(f"发布包目录={root}")
    print(f"torch={torch.__version__}")
    print(f"numpy={numpy.__version__}")
    print("发布包完整性检查通过")


if __name__ == "__main__":
    main()

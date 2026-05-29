# 文件清单

## 代码

- `run_contamination_aware_experiment.py`：完整训练和测试流程。
- `evaluate_existing_checkpoints.py`：加载已有 checkpoint 进行评估。
- `generate_enhanced_analysis.py`：根据结果 CSV 生成增强分析表格、曲线图和案例分析。
- `helpers.py`：路径解析、数据划分、指标记录和数据集统计工具。
- `src/dna_data.py`：CNR 数据读取、污染 reads 构造、read weighting 和图构建。
- `src/models.py`：Edge-aware MPNN 和 read auxiliary head。
- `src/train_eval.py`：训练、边分类指标、污染 read 识别指标和重构结果汇总。
- `src/decode.py`：图解码和序列重构方法。
- `setup_env.ps1` / `setup_env.sh`：创建本地 `.venv`。
- `run_quick_check.ps1`：小规模复评检查。
- `run_evaluate_existing.ps1` / `run_evaluate_existing.sh`：复评 1000 训练簇模型。
- `run_enhanced_analysis.ps1` / `run_enhanced_analysis.sh`：重新生成增强分析。
- `run_full_experiment.ps1`：重新训练并运行完整实验。
- `verify_release.py`：检查关键代码、数据、模型和结果文件是否齐全。

## 数据

- `data/raw/clustered-nanopore-reads-dataset/Centers.txt`
- `data/raw/clustered-nanopore-reads-dataset/Clusters.txt`
- `data/raw/clustered-nanopore-reads-dataset/LICENSE`
- `data/raw/clustered-nanopore-reads-dataset/README.md`

## Checkpoints

- `checkpoints/contamination_aware_train1000_test300_with_clean/read_weighted_mpnn_sim_pretrain_best.pt`
- `checkpoints/contamination_aware_train1000_test300_with_clean/read_weighted_mpnn_transfer_best.pt`
- `checkpoints/contamination_aware_train1000_test300_with_clean/read_weighted_mpnn_readaux_sim_pretrain_best.pt`
- `checkpoints/contamination_aware_train1000_test300_with_clean/read_weighted_mpnn_readaux_transfer_best.pt`
- `checkpoints/contamination_aware_train100_test50_with_clean/read_weighted_mpnn_sim_pretrain_best.pt`
- `checkpoints/contamination_aware_train100_test50_with_clean/read_weighted_mpnn_transfer_best.pt`
- `checkpoints/contamination_aware_train100_test50_with_clean/read_weighted_mpnn_readaux_sim_pretrain_best.pt`
- `checkpoints/contamination_aware_train100_test50_with_clean/read_weighted_mpnn_readaux_transfer_best.pt`

## 结果

- `results/contamination_aware_train1000_test300_with_clean/`：1000 训练簇、300 测试簇主实验结果。
- `results/contamination_aware_train100_test50_with_clean/`：100 训练簇、50 测试簇快速对比结果。
- `results/contamination_aware_contam40_test50_existing/`：40% 污染率补充测试结果。
- `results/train100_model_on_same300test_as_train1000/`：100 训练簇模型在 300 个测试簇上的复评结果。
- `results/train1000_model_reeval_same300test/`：1000 训练簇模型在同一批 300 个测试簇上的复评结果。
- `results/training_scale_same300_comparison.csv`：训练规模对比汇总表。
- `results/enhanced_analysis/analysis_report.md`：增强分析说明。
- `results/enhanced_analysis/tables/`：SRR、BRR、AvgED 主表，消融表，ReadAux 分析表，训练规模表和案例表。
- `results/enhanced_analysis/figures/`：SRR、AvgED、Edge-F1、Read-F1、平均边数图和训练规模对比图。

## 环境

- `.venv` 不进入版本管理。
- `requirements.txt` 保存除 PyTorch 之外的 Python 依赖。
- `setup_env.ps1` 和 `setup_env.sh` 会安装 CPU 版 PyTorch，并根据 `requirements.txt` 安装其余依赖。

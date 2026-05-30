# 污染感知 DNA 存储序列重构实验

本项目围绕污染读段簇条件下的 DNA 存储序列重构问题展开，主要实现了 read-weighted de Bruijn 图构建、插入/删除错误感知边结构、Edge-aware MPNN 边判别模型以及 ReadAux 污染读段识别辅助任务。项目包含实验代码、CNR 数据、训练得到的模型参数、主要实验结果和结果分析脚本。

## 项目内容

- `src/`：数据处理、图构建、模型定义、训练评估和解码模块。
- `data/raw/clustered-nanopore-reads-dataset/`：实验使用的 Microsoft CNR 数据。
- `checkpoints/`：训练得到的模型参数。
- `results/`：主实验、补充实验、训练规模对比和增强分析结果。
- `run_contamination_aware_experiment.py`：完整训练与测试流程。
- `evaluate_existing_checkpoints.py`：加载已有模型参数进行复评。
- `generate_enhanced_analysis.py`：生成表格、曲线图和案例分析。
- `setup_env.ps1`、`setup_env.sh`：创建本地 Python 虚拟环境。

## 目录结构

```text
.
|-- data/
|   `-- raw/
|       `-- clustered-nanopore-reads-dataset/
|-- checkpoints/
|   |-- contamination_aware_train1000_test300_with_clean/
|   `-- contamination_aware_train100_test50_with_clean/
|-- results/
|   |-- contamination_aware_train1000_test300_with_clean/
|   |-- contamination_aware_train100_test50_with_clean/
|   |-- contamination_aware_contam40_test50_existing/
|   |-- train100_model_on_same300test_as_train1000/
|   |-- train1000_model_reeval_same300test/
|   `-- enhanced_analysis/
|-- src/
|-- run_contamination_aware_experiment.py
|-- evaluate_existing_checkpoints.py
|-- generate_enhanced_analysis.py
|-- requirements.txt
|-- setup_env.ps1
|-- setup_env.sh
|-- run_quick_check.ps1
|-- run_evaluate_existing.ps1
|-- run_evaluate_existing.sh
|-- run_enhanced_analysis.ps1
|-- run_enhanced_analysis.sh
`-- run_full_experiment.ps1
```

## 环境配置

已验证的 Python 版本：`Python 3.12.13`。

Windows PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_env.ps1
```

Linux 或 macOS：

```bash
chmod +x setup_env.sh run_evaluate_existing.sh run_enhanced_analysis.sh
./setup_env.sh
```

环境脚本会创建 `.venv`，安装 CPU 版 PyTorch 和其余依赖，并运行完整性检查。由于虚拟环境包含本机路径和平台相关二进制文件，项目中只保留依赖文件和环境创建脚本。

## 快速检查

完成环境配置后，可以先运行小规模检查：

```powershell
.\run_quick_check.ps1
```

该脚本会加载已有模型参数，在 20 个测试簇上评估 0% 和 30% 污染率，用于检查代码、数据和模型文件是否可以正常使用。

## 复现实验结果

复评主实验模型：

```powershell
.\run_evaluate_existing.ps1
```

重新生成增强分析表格和曲线图：

```powershell
.\run_enhanced_analysis.ps1
```

重新训练并运行完整实验：

```powershell
.\run_full_experiment.ps1
```

在 CPU 环境下，1000 训练簇完整训练与测试约需 151 分钟。实际耗时会受到处理器性能、磁盘读写速度和 Python 环境影响。

## 训练规模对比

训练规模影响实验使用同一批 300 个测试簇，分别加载 100 训练簇模型和 1000 训练簇模型进行评估。相关文件如下：

- `checkpoints/contamination_aware_train100_test50_with_clean/`
- `checkpoints/contamination_aware_train1000_test300_with_clean/`
- `results/train100_model_on_same300test_as_train1000/`
- `results/train1000_model_reeval_same300test/`
- `results/training_scale_same300_comparison.csv`

如需重新评估 100 训练簇模型，可运行：

```powershell
.\.venv\Scripts\python.exe evaluate_existing_checkpoints.py --checkpoint-dir checkpoints/contamination_aware_train100_test50_with_clean --results-dir results/train100_model_on_same300test_as_train1000
```

## 大规模主实验结果

表中每个单元格格式为 `SRR / BRR / AvgED`。SRR 表示完整序列恢复率，BRR 表示碱基层面恢复率，AvgED 表示平均编辑距离。

| 污染率 | Unweighted Count Beam | Read-weighted Count Beam | Read-weighted MPNN Beam | MPNN+ReadAux Beam |
|---:|---:|---:|---:|---:|
| 0% | 0.180 / 0.958 / 4.573 | 0.415 / 0.967 / 3.632 | 0.916 / 0.995 / 0.595 | 0.926 / 0.995 / 0.582 |
| 5% | 0.200 / 0.964 / 4.007 | 0.415 / 0.960 / 4.385 | 0.913 / 0.991 / 0.963 | 0.913 / 0.991 / 0.967 |
| 10% | 0.233 / 0.958 / 4.657 | 0.465 / 0.955 / 4.963 | 0.930 / 0.990 / 1.060 | 0.923 / 0.993 / 0.823 |
| 20% | 0.263 / 0.964 / 3.953 | 0.498 / 0.953 / 5.207 | 0.880 / 0.979 / 2.338 | 0.876 / 0.977 / 2.522 |
| 30% | 0.291 / 0.931 / 7.602 | 0.538 / 0.962 / 4.171 | 0.856 / 0.974 / 2.819 | 0.856 / 0.979 / 2.321 |

## 结果文件

- `results/contamination_aware_train1000_test300_with_clean/method_summary.csv`：主实验结果汇总。
- `results/contamination_aware_train1000_test300_with_clean/read_contamination_metrics.csv`：污染读段识别指标。
- `results/enhanced_analysis/tables/`：SRR、BRR、AvgED 主表，read-weighted graph 消融表，Edge-aware MPNN 消融表，ReadAux 分析表，训练规模对比表和案例分析表。
- `results/enhanced_analysis/figures/`：污染率-SRR、污染率-AvgED、污染率-Edge-F1、污染率-Read-F1、平均边数和训练规模对比图。
- `results/enhanced_analysis/case_studies.md`：代表性簇案例分析。

## 数据说明

实验数据来自 Microsoft Clustered Nanopore Reads 数据集。数据目录中保留了原始 license 和说明文件。使用该数据集时应遵守其原始许可要求。

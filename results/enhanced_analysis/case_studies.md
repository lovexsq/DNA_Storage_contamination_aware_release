# 代表性簇案例分析

本文件选择少量典型簇，观察不同方法在同一簇上的错误差异。这里的 bridge/bypass 判断来自编辑差异的启发式分析，主要用于辅助说明错误形态。

## Count Beam 失败但 MPNN 成功

- cluster_id：3551
- 污染率：30%
- 参考序列长度：110
- 主要失败方法：Read-weighted Count Beam，ED=87
- 主要改进方法：Read-weighted MPNN Beam，ED=0
- 错误集中位置：错误主要位于参考序列第 1-6、8-14、16-26、28-32、... 位附近；替换 0 个、删除 87 个、插入 0 个。
- bridge/bypass 判断：错误以删除为主，可能存在绕过真实片段的 bypass 类风险。
- 污染 reads 支持判断：平均 read_weight 较低，说明簇内 reads 可靠性分化明显，错误路径存在被污染 reads 支持的风险。

## Count Beam 失败但 ReadAux 成功

- cluster_id：2218
- 污染率：30%
- 参考序列长度：110
- 主要失败方法：Read-weighted Count Beam，ED=84
- 主要改进方法：Read-weighted MPNN+ReadAux Beam，ED=0
- 错误集中位置：错误主要位于参考序列第 1-6、10-13、15、17-27、... 位附近；替换 0 个、删除 84 个、插入 0 个。
- bridge/bypass 判断：错误以删除为主，可能存在绕过真实片段的 bypass 类风险。
- 污染 reads 支持判断：平均 read_weight 较低，说明簇内 reads 可靠性分化明显，错误路径存在被污染 reads 支持的风险。

## MPNN 未完全正确但 ReadAux 降低 AvgED

- cluster_id：2400
- 污染率：30%
- 参考序列长度：110
- 主要失败方法：Read-weighted MPNN Beam，ED=75
- 主要改进方法：Read-weighted MPNN+ReadAux Beam，ED=2
- 错误集中位置：错误主要位于参考序列第 24-48、51-58、60-67、69-80、... 位附近；替换 0 个、删除 75 个、插入 0 个。
- bridge/bypass 判断：错误以删除为主，可能存在绕过真实片段的 bypass 类风险。
- 污染 reads 支持判断：平均 read_weight 较低，说明簇内 reads 可靠性分化明显，错误路径存在被污染 reads 支持的风险。

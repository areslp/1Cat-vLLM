# Run4 只读结果交叉复核

只复核已闭合数据；未改冻结分析、控制、渲染器或生产。读取的 RESULT SHA256 为 `1e6faaaa0edebd923c22d2de034fd9ab52ad6c2aa928870bfc53c0f94c8ba8fc`，render_results.py 为 `ae3ca5459fc4c90bf1221d765efea478460483b3df50e2e779e8d5539e78d4bf`。

1. **收益符号和 A/A 漂移正确。** `context-qsa-prep1/analyze.py:80–88` 定义收益 `100×(1−B/mean(A0,A2))`、漂移 `100×abs(A2−A0)/mean(A0,A2)`；逐行重算与 `rows[].common_decode.improvement_pct_per_pair/AA_drift_pct_per_pair` 相同。`render_results.py:29,35–45` 忠实呈现这两项；三臂表格值是每臂两次 pooled 指标的中位数，不是逐请求 ITL 中位数。主指标来自 `context_metrics.py:79–92` 的完整公共区间覆盖秒数/对应输出 token 数，属于客户端 SSE 观察，不能解释为 GPU 时间。`n_per_row=2`、`CI=null`，旧 NO_GO 未改变。

2. **c4 的六个可比较长度两次方向均正，但不等于准入。** 512/2K/8K/32K/64K/128K 的 `common_decode.improvement_pct_per_pair` 均为正。32K/c4 为 `+0.602/+0.776%`、A/A 漂移 `0.256/0.254%`；64K/c4 为 `+0.652/+0.578%`、漂移 `0.264/0.217%`，这两行的全部对应请求输出相等。8K/c4 为 `+0.526/+0.573%`，但 repeat1 有输出差异。128K/c4 为 `+0.363/+1.019%`，repeat0 漂移 `0.533%` 大于该次收益。保持全部样本，不能从 n2 建立稳定收益或扩大数值验收。

3. **c8 及回退路径限制必须保留。** 512/c8 和 2K/c8 两次 pooled 方向为正，但输出状态受 A/A 自差和下述逐请求 B 差异限制；8K/c8 为 `+0.736/−1.362%`，32K/c8 为 `−0.397/+0.563%`，两次变号。`candidate_route_expected='fallback'` 的 c1，以及近256K/c1、c2，只提供兼容性/原运行波动观察；`dynamic_candidate_activation='UNVERIFIED_NO_ADDED_DEVICE_COUNTER'` 不能被静态 c4/c8 挂接代替。

4. **整组 token 分类不能代表逐请求无 B 差异。** `analyze.py:63–67` 和 `audit_run4.py:276` 先比较整组，故其他请求的 A/A 自差会遮蔽某请求的 A0=A2、B不同。独立解码438条 main raw SSE（112128个输出 token），逐条与 GROUP token 验证后，146请求三元组为 **88全相等、48 A/A 自差、10 A0=A2且B不同**。10条为：512/c8 repeat0 index4；repeat1 index0/2/3；2K/c4 repeat0 index1/3；8K/c4 repeat1 index1；8K/c8 repeat0 index1、repeat1 index1/3（均0-based）。首差 token、原始文件路径/size/SHA 见 `PER-REQUEST-CROSS-REVIEW.json`。冻结整组27 PASS/11自差、exit3保留；不能把一次 A/A 相同直接归因为 QSA。`render_results.py:40–45` 是忠实复制整组标签，建议人审时注明“整组”并同时引用逐请求补充。

5. **64K/c8 缺失准确且有明确原因。** `row_id='context-65536-greedy-c8'` 的 `common_statuses` 六项均为 `NO_COMMON_WINDOW`，`common_decode.status='NO_COMPLETE_COMMON_COMPARISON_NO_FILTERING'`、六个 paired 值均 null。渲染器 `:18,38–44` 显示“缺失”，没有以零或筛选均值补齐；人审应写明公共 decode 窗口不存在，不能误解为请求失败或零收益。TTFT/端到端指标仍单独存在，不能替代该行稳态 decode 比较。

独立 raw 复核实际 exit0；非新增测试。脚本 SHA256 `b07163196ab1da47b3a6130ba70aec1b66fda9cdd414956ee3c2fd62ab966d74`，结果 SHA256 `c0b160fbf4bd8ec72a41ba97ff05791dcd2f2eba50656f0dce27df8b1ba802c2`。本轮窗口/outer1、冻结分析3、独立数据审计0和完整53/17/24恢复0是不同层级，不能合成整体 PASS。

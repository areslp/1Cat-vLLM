# QSA 优化：结果与独立 review 交接
日期：2026-10-03（America/New_York）

## 当前处置
本轮结束并封存，生产已恢复，完成复核的 heartbeat 已暂停。当前建议是暂停继续投入 QSA 的整轮服务实验，先交独立 review；不是认定所有 QSA 思路不可行，也不是宣称性能已经达到物理极限。没有永久上线、push 或合并生产分支。

本文件是封存后的新增交接说明；不改动旧代码、原始数据、退出码或归档清单。可转发指令见 [REVIEW-PROMPT-20261003.md](REVIEW-PROMPT-20261003.md)。

## 一页结果
|问题|已验证的结果|解释边界|
|---|---|---|
|覆盖|7档输入长度：512、2048、8192、32768、65536、131072、261632；19行×2次×3臂=114组，798 HTTP，112488输出token|128K/c8与近256K/c4/c8因KV容量事前排除；各长度未采用同一最大并发|
|实验结构|A0/off → B/on → A2/off，同组同长度统一释放|没有人为插入混长/staggered请求；仍不能据此假定GPU执行计划或动态batch完全相同|
|主要收益迹象|32K/c4：+0.602%、+0.776%；64K/c4：+0.652%、+0.578%，对应输出均相等|是B相对两份A均值的客户端共同解码窗口ms/token改善，不是已证实的GPU或整体吞吐收益；n=2、无CI|
|不稳定与缺失|8K/c8、32K/c8两次收益变号；64K/c8三臂两次共6组均NO_COMMON_WINDOW|不得删去缺失、填零或用TTFT替代共同解码指标|
|逐请求输出|146个主请求三元组：88三臂全同、48 A0≠A2、10 A0=A2但B不同|10例存在于有其他A/A差异的组内，原整组分类把它们归入AA_SELF_VARIATION；无法据此单独确定QSA因果|
|历史局部正确性|W1四张V100各280个有限样例，planner/原forward out与LSE、eager/graph等通过|不扩展为本轮所有上下文的logits、KV/GDN/PLE/draft状态逐位证明|
|资源|三臂client峰681353216/689508352/652791808B，1GiB门内，events.max/OOM/Swap为0；进程/设备采样最大32124/32165MiB，双门32384MiB|GPU是已保存采样，非连续峰值；静态c4/c8挂接不等于逐组动态命中|
|恢复|新鲜53请求、17门全部通过，24快0慢；恢复时API188448，Invocation30e0f4501bfe471c8d9871c8a2090322，随后health200|这是本轮结束的恢复检查点，后来的实时状态须重新读回|
|原退出码|分析3、window/outer1；原始数据审计与恢复退出0|数据完整、后处理成功、恢复成功都不等于实验通过|
|证据封存|36,398常规文件、3,083,389,890B、6条链接及926项输入/入口（924个plan输入+2个发布入口）已回读校验|文件完整性不是性能准入；本次新交接文件使用单独补充清单|

## 投入判断：建议暂时停在这里
目前可解释的收益很小，部分场景不稳定，输出差异也未解释。若继续为QSA做状态固定、扩展逐位验证、增加重复和维护服务挂接，成本可能超过这轮小幅收益的价值。因此建议保留成果、交独立review，暂不追加整轮A/B/A。

这是一项工程投入建议，不是统计结论或新的准入阈值。若reviewer能指出低成本、可复用的正确性定位方法，或有证据证明小幅收益在实际负载与规模上有价值，可以建议一次明确预算与停止条件的补充验证。不要只挑有利行，也不要因已有投入而默认继续。

“接近物理极限”的目标仍未证明达成。后续性能工作应先测主要耗时构成，再选方向。GPU kernel、内存访问、CPU/launch、调度等只是待测候选，现有证据不能确认哪个是最大瓶颈，也不能承诺还剩多少加速空间。

## Reviewer 阅读顺序
以下路径均相对主机证据根目录：
`/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930`

1. [完整review包](review/context-qsa-closure-20261002/REVIEW-PACK.md)；[19行结果和B分别对两个A的比较](review/context-qsa-closure-20261002/CONTEXT-RESULTS.md)。
2. [root独立逐请求复核](review/context-qsa-closure-20261002/ROOT-FINAL-REVIEW.json)与[另一实现的raw SSE复核](review/context-qsa-run4-postprocess-20261003/PER-REQUEST-CROSS-REVIEW.json)；[交叉意见](review/context-qsa-run4-postprocess-20261003/CROSS-REVIEW.md)。两份实现独立得到88/48/10。
3. [冻结分析](context-qsa-run4/analysis/RESULT.json)、[全量raw审计](review/context-qsa-run4-postprocess-20261003/RAW-AUDIT.json)、[冻结计划](context-qsa-run4/control/plan.json)和各臂 `control/attempt1/{A0,B,A2}/http/` 的原始记录。
4. [算法导航](review/context-qsa-independent-20261002/ALGORITHM-REVIEW-GUIDE.md)、[planner CUDA](implementation/source/planner58.cu)、[reference planner](implementation/reference54/source/planner54.cu)、[服务策略](service-implementation-retry2/policy.py)、[graph挂接](service-implementation-retry2/service_hook.py)。
5. [恢复53/17复算](review/context-qsa-run4-restoration-20261002/PARENT-RESTORATION-REVIEW.json)、[资源审计](review/context-qsa-run4-memory-review-20261002/actual-final-closed/MEMORY-REVIEW.json)、[封存回执](review/context-qsa-closure-20261002/ARCHIVE-READBACK.json)。

当前窗口 `context-qsa-run4` plan SHA256：
`355ef8caeee318fc79abad12cf4dcf70b05dd3a822177eb17b03ec2675b64ba4`
旧封存清单 SHA256：
`e5e6824e1cecc2c938859df35e204dfb98d9332e2b691170d18dc90cc6e1e24e`

## 必须保留的反例与限制
- 10个A0=A2/B不同的请求：512/c8 repeat0 index4，repeat1 index0/2/3；2K/c4 repeat0 index1/3；8K/c4 repeat1 index1；8K/c8 repeat0 index1、repeat1 index1/3。索引均从0开始，首个分歧token及raw路径在逐请求报告中。
- 原整组统计为27组全同、11组AA_SELF_VARIATION；不能据此说“没有B-only请求”，也不能把一对相同基线当作已证明确定性。
- c1–3回退，近256K/c1/c2仅回退兼容性；>32K为周期合成输入；prime后prefix hit可为0；未逐组证明候选动态命中。
- 历史service run4的主行INCONCLUSIVE、含mixed矩阵NO_GO仍保留，与本次context-qsa-run4分开。mixed调度问题不能直接替代同长度QSA评价。
- run1资源门、run2绑定、run3传输与客户端容量、CPU512容量等历史失败全部保留，见[封存前执行记录](review/context-qsa-closure-20261002/PRE-CLOSURE-REVIEW-PACK.md)，不拼接为本轮成功。
- 6条旧CPU fixture链接保留本机绝对路径，在Linux主机悬空；生成器stdout的旧0B快照和闭合181B副本均保留并说明。不要假设所有本机审计脚本可在主机原样运行。

## Review 期望输出与操作边界
请独立判断封存、仅一次有限定位、或继续开发哪种投入更合理，并解释依据。报告需含按重要性排序的发现、文件/行号或请求级证据、成立/不成立/未知的结论，以及任何建议验证的最小范围、成本上限与通过/停止条件。

review阶段只读现有实验和生产，可在新的独立目录新增审查报告；不要重跑GPU benchmark、重启/改服务、改候选或冻结证据、安装依赖、push或合并。若要复算，先核对脚本路径/环境/输出覆盖行为；主机Python使用既有 `/home/l/work/1Cat-vLLM/.venv/bin/python` 或仓库声明uv环境。本机Ruby补充审计不是主机已安装Ruby的保证。新实验须另行决定，不从本prompt自动启动。

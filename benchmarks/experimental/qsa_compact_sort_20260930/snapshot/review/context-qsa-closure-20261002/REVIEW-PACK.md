# QSA compact-sort：完整跨上下文结果与 review 交付

更新：2026-10-03（America/New_York）。本次窗口为 **context-qsa-run4**，与历史 service run4 是不同实验。当前结论：**数据矩阵完整、资源与生产恢复通过；输出一致性未通过整体验收，性能仅描述性，暂不准入。**

19 行、每行两次、三臂 A0/off → B/on → A2/off 全部结束。独立 raw 审计确认 **114 组、798 HTTP、112488 输出 token**，包括每臂 146 个主请求和 120 个单 token prime。A0/B/A2 HTTP 阶段各为 3003.25/3018.29/3032.44 秒（50.05/50.30/50.54 分钟）；三臂纯 HTTP 测量合计约 151 分钟，启动、预热、恢复另计。原窗口与外层退出 **1**，分析退出 **3**，全部保留，不能把收集完成或后处理 exit=0 写成实验通过。

## 首先看这些结果

- [19 行完整结果表及 B 分别对两个 A 的收益](CONTEXT-RESULTS.md)。每行的两次样本、A/A 漂移与缺失项均保留，无 CI 或临时准入阈值。
- [root 独立逐请求复核](ROOT-FINAL-REVIEW.json)和[可复算 Ruby 源码](root_review.rb)：解码 438 个主请求 raw SSE，核对完整 256 token 序列、GROUP/raw SHA，共 560 个输入文件摘要；复核配对算术、独立 raw 指标、资源与 53/17 恢复收据。
- [Sol 子代理独立交叉审查](../context-qsa-run4-postprocess-20261003/CROSS-REVIEW.md)及[另一份 raw SSE 逐请求复算](../context-qsa-run4-postprocess-20261003/PER-REQUEST-CROSS-REVIEW.json)得到相同 88/48/10 计数。审查引用的是渲染器修改前的 SHA；最终渲染已补“原整组输出分类”标签、逐请求提示和 B 分别对两份 A 的收益。
- [全量原始 SSE 审计](../context-qsa-run4-postprocess-20261003/RAW-AUDIT.json)：114/798/112488，evidence_errors 为空，审计器 exit=0；原实验分析 exit=3 保持。[后处理完整收据](../context-qsa-run4-postprocess-20261003/POSTPROCESS-RESULT.json)。
- [三臂资源审计](../context-qsa-run4-memory-review-20261002/actual-final-closed/MEMORY-REVIEW.json)。
- [53 请求、17 门生产恢复独立复算](../context-qsa-run4-restoration-20261002/PARENT-RESTORATION-REVIEW.json)及[结束后实时健康读回](FINAL-LIVE-HEALTH.json)。

## 性能：有局部收益迹象，尚不能宣布普遍有效

指标是所有同组请求共同解码窗口内的客户端 SSE 每输出 token 间隔。正值表示 B 相对两份 A 均值更快，不能解释为 GPU kernel 加速或物理下界。

| 同长度负载 | 两次配对收益 | 两次 A/A 漂移 | 解释 |
|---|---|---|---|
| 512 / c4 | +0.993%、+1.368% | 0.213%、0.387% | 输出有差异，仅描述 |
| 512 / c8 | +1.003%、+1.000% | 0.235%、0.041% | 输出有差异，仅描述 |
| 2K / c4 | +0.801%、+1.634% | 0.307%、1.835% | 第二次漂移较大，且有输出差异 |
| 2K / c8 | +1.195%、+0.582% | 0.855%、0.313% | 输出有差异，仅描述 |
| 8K / c4 | +0.526%、+0.573% | 0.065%、0.095% | 第二次存在输出差异 |
| 32K / c4 | +0.602%、+0.776% | 0.256%、0.254% | 两次输出一致，适合作为后续受控复现候选 |
| 64K / c4 | +0.652%、+0.578% | 0.264%、0.217% | 两次输出一致，适合作为后续受控复现候选 |
| 128K / c4 | +0.363%、+1.019% | 0.533%、0.199% | 输出一致，第一次漂移大于均值收益 |
| 8K / c8 | +0.736%、−1.362% | 0.092%、0.006% | 两次方向相反，且有输出差异 |
| 32K / c8 | −0.397%、+0.563% | 0.050%、0.259% | 输出一致但两次方向相反 |
| 64K / c8 | 缺失、缺失 | 缺失、缺失 | 三臂两次共六组均 NO_COMMON_WINDOW；不剔除、不补算 |

c1–3 是回退路径；近 256K 的 c1/c2 仅验证此矩阵下的回退兼容性，不提供候选加速结论。完整 19 行见结果表。当前 n=2、没有逐组候选动态命中计数，不能据此宣称收益达标或达到物理极限。

## 输出：纠正整组分类掩盖逐请求分歧的问题

冻结分析按整组数组判断 A0 与 A2：38 个三臂组比较中，27 个整组全相同、11 个为 `INCONCLUSIVE_AA_SELF_VARIATION`，没有整组级 `FAIL_B_ONLY_OUTPUT_CHANGE`。这不等于没有个别请求的 B 分歧。

root 直接从原始 SSE 独立提取 token，以 row/repeat/request_index 对齐 **146 个主请求三元组**：

| 逐请求关系 | 数量 |
|---|---:|
| A0 = B = A2 | 88 |
| A0 ≠ A2（基线自身变化） | 48 |
| A0 = A2，但 B 不同 | 10 |

后一类分布在 512/c8（4 个）、2K/c4（2 个）、8K/c4（1 个）、8K/c8（3 个）。具体 repeat、request_index、首个分歧 token 索引/值以及三个 GROUP 路径都在 `ROOT-FINAL-REVIEW.json/output_disagreements`。例如 512/c8 repeat0 的 request4，在第 5 个零基索引 token，A0/A2 为 20480，B 为 7936。

原分析与独立审计器共享整组分类语义；两者的原源码、结果、exit=3 均不改，逐请求补充是独立审查附件。**“10 个 A0=A2/B 不同”说明本轮请求输出不一致，尚不能证明 QSA 因果**：同组存在基线变化，尚未固定或证明 GPU 执行计划、batch composition、缓存/状态身份与逐组动态候选命中。也不能把这些差异一概归咎于用户要求排除的 mixed/staggered 调度。本轮没有人为插入混长请求。

历史 W1 每张 V100 各 280 个有限样例的 planner/原 forward out/LSE、eager/graph 检查仍是有效的有限证据，但不能扩展成这些完整服务上下文的 logits/KV/GDN/PLE/draft 状态逐位证明。

## 资源及恢复

三臂实际 client memory.max 都是 1073741824B；峰值 A0 **681353216B**、B **689508352B**、A2 **652791808B**，memory.events.max/oom/oom_kill 和 Swap 均为 0，CPU14/42 约束通过。模型 120GiB、KV663816、原始资源身份契约保持。

每臂四卡的已保存进程采样最高 **32124MiB**，设备采样最高 **32165MiB**，均低于两个 **32384MiB** 门。每臂 636 对 raw NVML/classification 记录保持 UNMAPPED，不伪造形状映射；另有已保存身份边界用于明确同 row/repeat/phase 的比较。单个最大值相同不等于候选新增显存为零，也不是连续峰值证明。实际 memory.stat、原始计数和缺失映射均保留。

当前生产 API **188448**，Invocation **30e0f4501bfe471c8d9871c8a2090322**，workers **189047–189050**。fresh 53 请求、17 项门全部通过，24 快/0 慢；G1 三轮中位数 32.2438/32.2885/32.2743ms，N1 为 33.0254/33.0040/33.0102ms。原 source/SO/shim/unit、实际 API flags、KV、四 rank E7、Swap/OOM/restart 与临时 model/client/timer/outer 清理通过。后续实时 `systemctl` 仍为同一 PID/Invocation、active/running、NRestarts=0，`/health` HTTP200。

原 raw child flags=false 的观测限制仍保留，实际 API 配置由单独只读 /proc 采集核对。恢复的成功与实验非零退出分别报告：[原 launcher 实际回执](ROOT-LAUNCH-RECEIPT.json)为 OUTER_EXIT=1 / POST_OUTER_CLEANUP_EXIT=0。

## 实验设计与冻结身份

512、2K、8K、32K、64K：c1/c4/c8；128K：c1/c4；261632 tokens：c1/c2。128K/c8 和近256K/c4/c8 因名义 KV 容量在实验前排除。每组同长度统一释放、不插入新的混长请求。512 为冷请求；其余先做相同 prompt/salt 的单 token prime，实际 prefix hit 可以为零。>32K 是明确的周期合成输入，不代表自然长文分布。各请求 TTFT、prime 冷 TTFT 和完整请求耗时另存原分析，不能与共同窗口解码指标互换。

- plan SHA256：`355ef8caeee318fc79abad12cf4dcf70b05dd3a822177eb17b03ec2675b64ba4`
- control MANIFEST：`31efd893097bf3768f9d6b958143f82b59e59cf1e8efa251189875d6fb15a079`
- [启动前独立复核](../context-qsa-run4-execution-20261002/ROOT-EXECUTION-REVIEW.json)：597 项现场 plan 输入、323 项新包/staged SHA、fresh baseline 与恢复适配。
- [CUDA planner](../../implementation/source/planner58.cu)、[绑定](../../implementation/source/bindings.cpp)、[服务策略](../../service-implementation-retry2/policy.py)、[graph 挂接](../../service-implementation-retry2/service_hook.py)、[算法阅读导航](../context-qsa-independent-20261002/ALGORITHM-REVIEW-GUIDE.md)。
- [历史执行和失败全记录](PRE-CLOSURE-REVIEW-PACK.md)：run1 的旧 process 门失败、资源诊断、run2 的 binding 失败、run3 的 512KiB 流式单行与512MiB客户端容量失败，以及真实 CPU512 FAIL/CPU1024 PASS 均保留，未倒填或拼接进本轮。

运行环境为本机 uv 托管 Python3.12.13，远端既有 `/home/l/work/1Cat-vLLM/.venv/bin/python` Python3.12.13 / uv0.11.16 / requests2.34.2 / urllib3 2.7.0；root 补充审核使用 macOS Ruby2.6，仅标准库。没有安装或替换运行环境。

## 给 reviewer 的下一步

1. 先审查逐请求分歧和整组聚合的遮蔽问题；将相同 A0/A2 但不同 B 的请求作为定位样本，避免以整组 AA_SELF_VARIATION 掩盖它们。
2. 在后续独立窗口中，从这些确切样本固定执行状态与 batch/graph 路径，记录候选实际命中，在首个分歧附近对 planner 描述符和原 forward 输出/logits 逐位定位。先确认可重复的原版自变范围，再归因 QSA。此轮没有自行追加该实验。
3. 输出解释清楚后，优先复现 32K/c4、64K/c4 的小幅收益；增加事前登记的重复数与顺序控制，核对 c1 回退路径的噪声/额外开销。8K/c8、32K/c8 的变号应保留为不稳定证据。
4. “物理极限”还缺硬件计数、kernel 时间分解、HBM 流量/带宽与可比下界；本轮 SSE 不能回答该问题。历史 mixed NO_GO 与原服务主行 CI 不足继续保持，另归调度审查，不覆盖本轮同长度结果。

## 证据保存和退出状态

主机根目录：`/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930`；本地镜像：`/private/tmp/flash-next-step58-20260930`。本页、结果表、源码、原始数据、所有失败收据与复核均留在主机供人审查。

子代理首归档核对 14474 个 pins；其归档生成器 stdout 当时仍在写，快照误收 0B 的问题已明确记入 [ARCHIVE-SNAPSHOT-ERRATUM.json](../context-qsa-run4-postprocess-20261003/ARCHIVE-SNAPSHOT-ERRATUM.json)，闭合的181B原文另保存在 `transport-closed/` 并通过补充校验；没有改写原始 benchmark 数据或旧快照。最终 root 清单与回读结果见 `ARCHIVE-MANIFEST.json`、`ARCHIVE-READBACK.json`；发布入口回执见 `PUBLISH-RESULT.json`，这些是文件完整性证明，不是性能通过。

完成事件已实际触发本次复核，`qsa-benchmark` heartbeat 已暂停，见 [PAUSE-RECEIPT.json](../context-qsa-run4-monitor-20261003/PAUSE-RECEIPT.json)。没有永久上线、push 或生产分支合并。

归档预检发现历史 CPU fixture 的软链接包含原本机绝对路径，在 Linux 主机上可为悬空链接。原链接不改，最终清单分别记录并回读校验链接文本及目标存在状态，常规文件和 plan 钉住的实际输入单独校验。见 `SYMLINK-INVENTORY.json`、`PREFLIGHT-FAILURES.json`；不把这些 fixture 链接描述为可在主机直接重放的环境。

[独立归档范围审查](../context-qsa-run4-postprocess-20261003/ARCHIVE-SCOPE-REVIEW.json)一次 SSH 核对：41 个目录两端存在，本机常规文件无漏传，五个冻结 plan 的 924 个不同输入全部 SHA 符合且无跨版本冲突。6 条 CPU fixture 链接文本一致、主机均悬空。唯一已披露的大小差异为归档生成器 stdout 的 0B 旧快照与181B闭合副本。run2 memory 的5件材料是实有 CPU/source 准备，因该轮零 HTTP 而没有运行结果；没有补造空报告。隔离 provider CLI 的 codex-home 状态不上传，模型/请求/返回/验证证据已经单独保存。

# STEP58 service 单次源码纠正

本包是service-implementation的隔离纠正版本；原82文件及window1失败不可变。父级已审最小源码差异，未授权本包启动服务/GPU。当前恢复结论由control持有，本源码包不代替恢复验收。

真实window1错误是四rank在nativewarmup构造SchedulerOutput后进入observer，`set(None)`抛TypeError；不是candidate planner数值分歧。active `source/scheduler_output.production.py`219–266明确preempted_req_ids允许None且make_empty省略该字段。warmup.py305–310/344–358/361–363构造prefill/decode/cleanup并调用worker_execute_model，未传dummy_run=True。gpu_worker.py846–848在capture之后调用nativewarmup，875–885才完成warmup/preload；故capture-ready不能当最终warmup完成的生命周期重置点。

修复限定为：

1. preempted None按原finish_requests的空集合语义处理；finished_req_ids仍是必填set，其他坏类型继续failclosed。
2. 首个严格ID绑定client epoch清startup slot maps/ledger一次；首epoch前的warmup不消耗lifecycle事件预算。reset数量随drain保存。其后真实零token完成/删除仍被记录。
3. 所有slot add/append仍保留实际owner/初始物理映射，即使该owner是sentinel或未归因请求；不跳过中间占用，不伪造primary-A直接复用primary-B。
4. reuse旧owner必须属于已完成HTTP200、n=1/singleprompt、queue_drained且严格external/internal绑定的primary client receipt。predeclared名单不够；sentinel和未归因warmup不能成为旧owner。admission CLI读取此前真实drain/receipt并校验rank/PID/epoch名单，接口/状态不变。

CPU attempt1 exit0，22/22 PASS（6新增回归、5protocol、11service），`evidence/cpu-retry-attempt1.log`。新测试执行active SchedulerOutput dataclass AST/make_empty和实际warmup赋值/调用片段，穿过Runtime.install wrapper及model_runner首块/finish原AST；其模型/GPU主体是明确未执行的CPU stub。旧冻结wrapper在同样调用路径复现NoneType错误。覆盖非空preemption、首epochreset、真实零tokenfinished、completed-primary门、intervening-untracked反例及坏类型failclosed。不是完整CLI/model startup或服务数值PASS。

candidate.so、原prod-w48浮点forward、E7/MTP4/guard/P2P、graph域/owner/helper/tail、shadow/firstbank/smoke代码未改。已成功的window1 synthetic bank CUDA graph smoke只按相同源码/二进制指纹复用；不重跑它，也不把synthetic PASS升级为真实服务PASS。指纹对照及原smoke receipt见 `evidence/unchanged-mechanism.json`。

新源入口为D58/service-implementation-retry1；父级controller另冻新的输出路径/config/有限窗口和到期恢复。历史SOURCE-REVIEW.md、evidence/review-packet-v1及旧CPU-TORCH证据随原82保留，不代表当前retry新增验证。新权威合同为contract.json及command-contract.json，源码差异为evidence/retry-diff.patch；完整新清单manifest.json。

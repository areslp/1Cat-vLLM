# STEP58 host event整型边界纠正：retry2源码包

状态：CPU/source PASS，未执行retry2服务、GPU、HTTP、候选准入或W2。继承`RETRY-SOURCE-REVIEW.md`只描述retry1 warmup修复，本文件才是本次权威delta说明。old82/retry1/run2全部保留不可变。

真实run2已通过nativewarmup/HTTP/four-rank source gate/360s成熟，c1 primary完成16token；独立sentinel在四rank drain JSON编码时因NumPy int32失败。源链是Eagle `num_scheduled_tokens.max()`→`get_uniform_token_count`原样返回→DP1 passthrough→other-manager dispatch event.uniform。不是新planner/浮点分叉，shadow未开始。原运行并无partial drain JSON：io.save在open(xb)之前做json.dumps。

唯一运行代码delta为`Runtime.event`三行：ACTIVE/audit guard和4096cap之后，导入新增`event_contract.normalize_event`再append。它重新构造独立host record，严格审计四种既有producer，绝不改dispatch入参、返回descriptor、scheduler或tensor。

| schema | 被审整数字段/规则 |
| --- | --- |
| dispatch | actual_requests 1..8；actual_tokens 1..INT32_MAX；uniform optionalNone否则1..INT32_MAX；selected tokens同前，requests optionalNone否则1..8，uniform optionalNone否则正整数，bucket optionalNone否则0..INT32_MAX；mode保留NONE/PIECEWISE/FULL名字；role target/other |
| real_scheduler | exact unique request_ids；scheduled_tokens exact同ID集合/正整数；draft_counts仅该集合子集/0..4；sentinel必须Python bool |
| original_owner_call | owner字符串；target/draft role；exact original_no_target_capture_ticket route；无整数 |
| unsupported_original | exact三元素descriptor与三元素q_shape；各正整数；不扩大候选unsupported route |

整数字段仅接受**exact Python int**或**NumPy integer scalar**，无损转为Python int；拒绝Python/NumPy bool、float、array、tensor、int子类、custom __int__。None仅明确optional字段可用。未知event/未知字段failclosed。没有global default=str/default=int、没有GPU .item/read或宽松encoder。

bank/io_contract/shadow/smoke/policy及其它14个核心运行文件全部与retry1逐字相同；candidate SO和原prod-w48 forward不变。原GPU bank smoke仅可按父/control精确原receipt/source/unit/resource绑定复用，本包不重跑、不把CPU stubs当真实shadow数值或graph PASS。

真实NumPy CPU evidence：

- attempt1：exit1，5tests/0failures/3errors。stage27依赖漏掉make_config.SOURCES所需原13源码中的部分文件，FileNotFound发生在fixture配置建立、尚未到event链；不能称此attempt已复现旧失败。完整log/exit/RESULT保留。
- 经父审原因，只补齐13source与38总依赖，不改运行代码/5tests，create-only attempt2：exit0，5/5 PASS，0errors/0failures。0.025s；uv0.11.16，现有Python3.12.13，NumPy2.3.5，torch未import，CUDA_VISIBLE_DEVICES为空，taskset14/nice15/OMP-MKL-OPENBLAS1，ru_maxrss44716KiB（实际RSS峰，不宣称cgroup硬限）。
- test确实编译active Eagle779/780/781/823/830、get_uniform、DP1、native descriptor/dispatch AST，真实NumPy int32沿源码进入实际安装的Runtime execute/sample/dispatch wrappers与event→原drain→未改io.save。旧retry1 TypeError被复现且drain文件未创建；旧fixture坏字段类型/值作为类型化JSON保留。
- 新off+shadow full drain JSON roundtrip通过，包含main+sentinel/IDbinding/epoch/events/counts/lifecycle字段。所有7dispatch整数leaf严格负例、None、上限/溢出和四producer/schema/noop/不alias检查通过。model/GPU replay/counter/allocator/synchronize明确为CPU stubs；没有真实HTTP或candidate attention。

补充源码快照为`source/eagle_speculator.production.py` SHA9cae2d54f70ed54ef1f5416014cc19862685e6d33ec7833a3fba629f048b8f61、`source/dp_utils.production.py` SHA1400a8cb2c37bfb0fb49efe9a26e26f56a18329248cdc9c365732a656115967a。只是此次证据/未来control可核source pin；原13runtime loader pins与stage source gate不扩展、不变更。

`freeze_retry2.py`从唯一canonical retry2根赋值CLI/cwd/shim，不做重复substring replacement。三个可执行CLI皆严格验证本地同名文件和精确root；canonicalization应用两次仍相同。继承retry1 manifest/command/contract和旧badfreeze保留为证据，最终入口仅本次新manifest/command。config examples仅review用，ID库存仍原296生成机制；未来实际run3/cohort新ID/输出路径/窗口gate由control另冻结，尚无服务授权。

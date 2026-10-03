# QSA compact-sort：跨上下文执行检查点

本页供代码与实验review使用。完整跨上下文窗口run3已在A0的128K首个prime停止：30/38组完成，流式单行上限及客户端实际内存门均失败，B/A2未运行；生产恢复已独立通过53请求/17门。先前run2因测试binding接线错误在0请求时停止，已独立完成53请求/17门恢复。此前run1 A0在2048输入、c8组后的显存门停止，B/A2没有运行；独立资源诊断复现四worker32058MiB超过旧process32000，整卡均32099MiB低于原device32384。诊断后新PID生产恢复已独立通过53请求/17门/24快0慢。新合同保留原整卡门；本页不批准永久上线。

## 已证明什么

|问题|结论与范围|
|---|---|
|QSA方案是否全不可行|不是。direct-scatter局部节省约0.060ms/c8未过筛选；compact-sort历史净局部节省约0.589–0.595ms/c8。|
|算子逐位正确性|历史W1四V100各280有限样例，整数规划、原forward out/LSE、eager/graph及输入/guard检查通过；不是任意输入的形式证明。|
|服务数值|历史B对两份对齐A的有限逐位检查通过；未证明完整KV/GDN/PLE/draft状态身份，也没有扩展到此次所有长度。|
|历史服务收益|run4主行约0.758%，CI下界0.446%未达旧0.5%门，INCONCLUSIVE；含mixed的完整矩阵NO_GO保持。|
|新跨上下文结果|run3计划7档长度、19行、114组、798请求；实际A0完成前30组至64K，在128K首prime失败，B/A2未运行。旧run1和独立资源诊断各完成62HTTP、11282输出；run2为0请求。不能拼接成新跨臂结论。|
|是否到物理极限|未证明。SSE是客户端观测，不是GPU执行时间、HBM流量或硬件下界。|

## 从这些代码开始

- [算法导航与具体函数行](../context-qsa-independent-20261002/ALGORITHM-REVIEW-GUIDE.md)：先读压紧、排序桶、相等key稳定顺序、padding/barrier与graph生命周期。
- [CUDA planner](../../implementation/source/planner58.cu) 与 [binding](../../implementation/source/bindings.cpp)，对照 [reference planner](../../implementation/reference54/source/planner54.cu)。
- [服务代理](../../service-implementation-retry2/policy.py)、[graph挂接](../../service-implementation-retry2/service_hook.py)、[原shim链](../../service-implementation-retry2/sitecustomize.py)。原浮点forward继续来自原SO；c1–3回退，c4–8仅特定FULL形状可进入候选。
- [W1检查](../../w1-implementation/gate.py)、[W1报告](../../W1-REPORT.md)、[有限数值双基线审查](../NUMERIC-B1-BOTH-REVIEW.json)。静态graph挂接不等于逐组动态命中证明。
- [本轮测量合同](../../context-qsa-prep1/CONTRACT.md)、[客户端](../../context-qsa-prep1/runner.py)、[指标](../../context-qsa-prep1/context_metrics.py)、[完整性与分析](../../context-qsa-prep1/analyze.py)、[独立raw复算器](../context-qsa-independent-20261002/independent_review.py)。

## 新实验怎样隔离问题

512、2K、8K、32K、64K使用c1/c4/c8，128K使用c1/c4，近256K（261632 tokens）使用c1/c2；三条超名义KV容量行事前排除。每行两次，A0/off→B/on→A2/off；三臂对应body仅request_id不同，salt相同；不同请求/组的salt独立。组内同长度、统一释放，不插入新的异长请求。>32K提示为明确的合成周期token负载，不声称自然长文代表性。

512是冷请求；其余每请求先用相同prompt/salt做1-token prime，再运行256输出。冷prime TTFT与主请求TTFT分开；实际prefix hit包括0，不因执行prime就声称热缓存。主指标只计所有请求共同窗口内完整positive-to-positive区间，以实际输出token数归一化；边界区间不插值。没有共同窗口时保留不足，不剔除。n=2只给描述性配对差和A/A漂移，不生成CI或性能准入。近256K的c1/c2仅回退兼容性。

## run1停止原因与缺证

[完整处置](RUN1-DISPOSITION.md)与[资源调查](../context-qsa-memory-failure-20261002/REPORT.md)保留全部失败。进程32000MiB门来自旧短负载31330MiB样本；另有独立整卡32384MiB门。本轮保存值从31330→31490→31834MiB，最后prime后各卡device31875MiB。失败瞬间raw未保存，既不能断定具体超出值，也不能断定OOM或整卡门被突破。所有旧门保持，不将成功HTTP覆盖资源失败。

优先review：已有shape相关增长的证据与未记录的失败原值；原baseline工作量需求与candidate新增占用如何分开；整卡裕量与KV663816是否同时满足。该阶段决定先补已有组外NVML查询的raw记录，重复同一有限片段。随后诊断取得下节原值；旧run1未保存的值保持未知，未用诊断样本倒填。

## 独立资源诊断：已经取得失败原值

`context-qsa-resource-run1` 只重放原A0前11组，同一body/salt/顺序，原资源门不变。第11组再次由进程显存门STOP：四worker153457–153460均32058MiB，超过32000MiB门58MiB；紧随的四UUID整卡查询均32099MiB，距原32384门285MiB、距32768物理总量669MiB。查询相隔数十毫秒，不是连续峰值记录，也不构成后续长上下文容量证明。原错误不是0/未知字段；未见OOM证据。旧run1和本诊断均保留FAIL，不事后改成PASS。

raw出处为 `context-qsa-resource-run1/control/attempt1/A0/http/nvml-raw/` 的 q0115 process与q0116 device。q0115 SHA `3f1279e917ca897bbfad417eb5a1c44e40476e8bf626b49c1a20632b91d37e7b`。客户端退出前实际cgroup峰364703744字节，512MiB门内，OOM/Swap均0；systemd CLI显示值不用于真实峰结论。独立SSE/资源审计另列于 `review/context-qsa-resource-review-20261002/`。

后续窗口已预登记process与device同为32384MiB，保留原整卡门、384MiB物理余量、KV663816、全部身份/CPU/主机内存/Swap/OOM约束，原798请求不变。理由是旧32000来自短负载样本而非硬件限制，已被相同原版的合法负载需求超过；新数据须使用独立window，不拼接失败样本。candidate新增显存仍需与同形状两个A采样比较，任何整卡越线仍STOP。该更改不证明已经到物理极限。

## run2执行绑定

资源决策已[独立审查](../context-qsa-resource-review-20261002/RESOURCE-CAP-DECISION.md)，[控制源码复核](../context-qsa-resource-review-20261002/RUN2-CONTROL-REVIEW.md)覆盖61文件、357依赖和仅scope替换的14恢复文件；旧共享guard未改，owned identity只改process常量。实际入口module/function及SHA进入现场binding。原数据矩阵、预算、分析和候选代码不变。

新窗口 `context-qsa-run2` 的plan SHA为 `b22269513dc30240aa1f88ca671dba5580410fad1a5a6b5708f9056480b7a3ba`，control MANIFEST为 `22c303f0cc87664e80a60e0d819fd3134650b28679c4992160a76415b6805951`。root复核497项现场plan输入、244项新包/staged本地SHA及6个fresh baseline门；[完整技术复核](../context-qsa-run2-execution-20261002/ROOT-EXECUTION-REVIEW.json)。三臂统一process/device32384，原32000合同的两次失败继续保留。新窗口结束须再次进行独立53/17恢复。

run2实际客户端在0.787秒后退出1，尚未创建任何请求：新增`resource_guard_identity`放进了只允许12个key的core binding，未修改的`Observation.__init__`严格拒绝第13个字段。startup此前只调用`actual_identity`，没有执行这个constructor；本机CPU和独立审查均漏掉了这条生产者/消费者接线。此失败不是QSA性能或数值结论，原SOURCE PASS收据与运行FAIL同时保留。

[主机真实文件诊断](../context-qsa-run2-execution-20261002/REAL-CONSTRUCTOR-DIAG.json)复现原错误；仅在RAM中移除此字段后，原constructor成功核对165实际source SHA/stat、4 capture SHA及startup SHA，未知额外字段仍被拒绝，无HTTP/GPU/服务变更。正式修复采用原12字段的生产者输出，resource proof放在SHA绑定的startup收据和原sourcepins中，startup发布binding前真实构造原Observation；不采用运行时过滤未知字段的变通。原run2全部失败证据保持，原798矩阵/32384双门不变。

## run2恢复与run3执行绑定

[run2恢复独立复算](../context-qsa-run2-restoration-20261002/PARENT-RESTORATION-REVIEW.json)确认新API159520、Invocation `5dab31dbb15240258543ca6613f5974b`、workers160034–160037：53请求、17门、24快0慢通过。三轮G1为32.3163/32.2779/32.2861ms，N1为32.9997/32.9978/32.9991ms；原source、KV、unit/API配置、四rank E7和临时服务清理通过。实验window/outer/SSH为1，恢复与清理均0；接口失败保留。

[prep3修复独立审查](../context-qsa-run2-independent-20261002/PREP3-FIX-REVIEW.md)通过。控制MANIFEST `e8c9432b2d45394f94d2a74d748da1ddfcec45e41596ffc0e793a2f203a926a0`，run3 plan SHA `52ca5b2f569ecd0096eb40c3ec596a6fd2118fd4a64d4044f381e585d9d49fa4`；root复核507项现场输入、244项新包/staged本地SHA及6个新基线门。原12字段binding由发布前真实constructor检查；资源metadata在SHA绑定startup receipt中。21项作者CPU检查与8项独立定向检查通过，不替代现场结果。原798矩阵/预算/候选/双32384门不变。

[run3独立审计入口](../context-qsa-run3-independent-20261002/GUIDE.md)将复算闭合的三臂SSE、共同窗口及输出一致性；[恢复适配审查](../context-qsa-run3-independent-20261002/RESTORE-ADAPTER-REVIEW.json)只证实路径和旧基线身份变化，现场53/17恢复仍须在本窗口结束后完成。

## run3已保存的双重失败

独立原始SSE审计复算30个闭合组、234个完整HTTP和33384个输出；闭合组没有审计错误。28组具有COMMON_WINDOW；64K/c8的两次均为NO_COMMON_WINDOW，不能用于QSA收益归因，缺失值不删除。原审计退出1明确保留失败prime，不因成功前缀将整臂视为通过。

A0于第31组 `context-131072-greedy-c1-repeat0/prime-00` 失败；前30组覆盖512至65536的c1/c4/c8各两次。client报告运行1902.03秒后STOP，window/outer恢复流程保留非零实验退出。

首个失败是transport的单行门：实际worker收到620725B后在43.34秒抛出cap错误，旧LINE_CAP为524288B，总量16MiB、line-count4096和600秒门未触发。检查在raw保存前执行，所以失败行未保存，raw文件为空；不能声称知道该行内容、首token时间或输出。已保存64K首帧约310502B，包含完整prompt_token_ids，支持检查128K/261632的合法响应容量，但不能补造失败行。

另有独立CPU客户端资源失败：[actual-stopped-A0/MEMORY-REVIEW.json](../context-qsa-run3-memory-review-20261002/actual-stopped-A0/MEMORY-REVIEW.json)复核实际cgroup peak537501696B，相比512MiB的536870912B多630784B；memory.events.max823、OOM/Swap0。memory.current331866112B是退出后读数；旧证据没有memory.stat，不能确定峰值由heap、子进程或page-cache哪个部分造成。systemd CLI的1.5M不是该cgroup实测峰。

同一独立解析已通过30组和540对raw/classification输入核验。四卡sampled process最大32124/32124/32122/32124MiB，device最大32165/32165/32163/32165MiB，记录到的GPU门未失败；这些都是采样边界，128K失败请求没有完整post边界，不是连续峰值或128K兼容性证明。B尚未运行，无B-A显存或性能比较。

下一步仅修测试接收与资源合同，候选代码保持。最长输入的真实loopback transport与native CPU cgroup容量需要先验收；旧run3不追认为PASS，下一完整窗口不拼接旧30组。

## 长上下文传输修复与独立CPU容量验证

`context-qsa-transport-prep1`只将流式单行上限从512KiB改为2MiB，并为拒收行保存完整SHA/长度及有界前缀；总量16MiB、请求树34MiB、packet4MiB、结果256KiB、4096行和原时限继续保留。原matrix、body/salt/顺序、输出数、解析与指标不改。原run3未保存的失败行仍未知。原Pump源逐字节不变，通过新控制层显式选择拥有新worker的相对路径。

主机真实requests/urllib3/socket/子进程进行完整A0顺序的模拟传输：38组、146主请求、120prime，另含3个拒收负例。CPU-only模拟HTTP服务使用独立临时端口及独立cgroup，不联系8200/8201。客户端提前写入旧A0原始HTTP树196829816B，用于保守容量负载；不将其当作旧run3峰值构成。

[512MiB实测](../context-qsa-run4-execution-20261002/CPU512-ROOT-REVIEW.json)：wire全部通过，但实际peak537616384B、events.max8762，容量FAIL。OOM/Swap0，失败原样保留。

[单次1GiB实测与root容量决定](../context-qsa-run4-execution-20261002/CPU1024-ROOT-REVIEW.json)：同五份源码、相同完整顺序与预写入，wire全部通过；实际peak890236928B，距离1GiB上限183504896B，events.max/OOM/Swap均0。生产API166245及Invocation未变，两个临时CPU单元均已清理。运行profile为主机既有Python3.12.13、requests2.34.2、urllib3 2.7.0。

下一完整run4的A0/B/A2统一使用1GiB客户端预算，明确覆盖原512MiB工程预算；模型120GiB、进程及整卡两项32384MiB门、KV663816、CPU14/42、原时限与798请求保持。客户端保存memory.stat并要求events.max=0。该CPU容量证明不代替GPU实测、服务收益或物理极限证明；run4已完成控制封包、fresh基线与root计划复核，并下发主机执行。

## 新完整run4执行绑定（进行中）

本轮名称是`context-qsa-run4`，与上文历史服务run4不是同一次实验。冻结plan SHA为`355ef8caeee318fc79abad12cf4dcf70b05dd3a822177eb17b03ec2675b64ba4`，control MANIFEST为`31efd893097bf3768f9d6b958143f82b59e59cf1e8efa251189875d6fb15a079`。root复核597项主机plan输入、323项新包/staged本地SHA、6项fresh baseline门和14份仅scope替换的恢复文件，见[独立启动审查](../context-qsa-run4-execution-20261002/ROOT-EXECUTION-REVIEW.json)。作者最终11项接线CPU检查与7项客户端预算检查通过；原constructor仍严格12字段，transport身份进入SHA绑定startup收据与实际每请求close记录。

计划仍是完整798HTTP/112488输出，不拼接run3已完成30组。当前只说明计划启动，尚无新跨臂收益或恢复结论。窗口结束后必须再取得fresh 53请求/17门恢复及独立raw复算，才完成交付。

## run3失败后的生产恢复

[新53请求独立复算](../context-qsa-run3-restoration-20261002/PARENT-RESTORATION-REVIEW.json)通过17门、24快0慢；API166245，Invocation `e3ca051687274273a7c61b0861841142`，workers166757–166760。三轮G1中位数32.2567/32.2678/32.2626ms，N1为32.9717/32.9487/32.9561ms。原HEAD/source/SO/shim/unit、实际API flags、KV663816、Swap/OOM/restart0、GPU归属和临时model/client/timer/outer清理均通过；raw child flags=false的观测限制继续保留。E7四rank一致，compressed60、eligible136、hook330、reused76。

实验window/outer/SSH为1；恢复verify/evaluate/postouter/artifact/cleanup/start为0。恢复证明与失败实验分别保存；下一次GPU窗口仍须重新验收。

## run1生产恢复与证据（诊断前检查点）

[独立恢复复算](../context-qsa-control-review-20261002/PARENT-RESTORATION-REVIEW.json)：53请求/17门通过，24fast/0slow，六轮约32.257–33.050ms，固定token/JSON logprob、fresh四rank E7通过。当前检查点API149598、Invocation `abb15cc7d96d486a973ee50b6039f148`、workers150107–150110。原HEAD、source/SO/shim/unit、实际API环境、KV663816、Swap/OOM/restart与GPU归属核验通过；raw child flags=false限制保留。

实验window/outer/SSH=1，恢复verify/evaluate/postouter=0，各自记录。资源诊断后必须另做新PID恢复，不能沿用此检查点；新适配见 `review/context-qsa-resource-restoration-20261002/`。

## 资源诊断后的新恢复检查点

[独立53请求复算](../context-qsa-resource-restoration-20261002/PARENT-RESTORATION-REVIEW.json)与[fresh只读现场身份](../context-qsa-resource-restoration-20261002/FINAL-ORIGINAL-READBACK.json)均通过。API155384、Invocation `35db2e486c754872a260c274e40ffab3`、workers155991–155994；17具名门全部true、24快0慢。三轮G1中位数32.2273/32.2615/32.2207ms，三轮N1为32.9669/32.9835/32.9686ms，逐请求和汇总均过原门。53完整输出、接受计数、fixed JSON logprob与fresh四rank E7复算通过。原HEAD/SO/shim/unit/实际API配置、KV663816、Swap/OOM/restart0、GPU归属和临时服务/timer/outer清理通过，raw child flags=false的历史限制保留。

本次window/outer/SSH仍为1，恢复verify/evaluate/postouter/artifact/cleanup/start均0。实验失败与恢复成功分别记录；此检查点只证明资源诊断已收尾，下一窗口结束仍须新的独立恢复。

本次候选SO SHA `e5ac0b418ebb9a387e0838b230dd15114185f637804e9d14b336897ab24bb7e3`；原forward SHA `a2c3c845572e36a6eb788e53c1861734285e2512ffb12bc5b3c19909871e433b`；run1 plan SHA `543f74a179ad93c9c51a52da55c921c7215b1a667af432399e133446ef5bb634`。所有代码与原始实验都位于主机 `/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930`，本地镜像位于 `/private/tmp/flash-next-step58-20260930`。

没有永久上线、push或生产分支合并。

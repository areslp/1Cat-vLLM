请独立 review oh-my-gpu 上的 QSA compact-sort 优化，判断现有证据是否可靠，以及继续投入是否值得。通过本机直接 ssh oh-my-gpu 访问。

证据根目录 D：
/home/l/work/flash-next/perf-20260923-resume/step58-qsa-compact-sort-20260930

先读 D/REVIEW-HANDOFF-20261003.md，再按其中导航检查：
- review/context-qsa-closure-20261002/REVIEW-PACK.md
- review/context-qsa-closure-20261002/CONTEXT-RESULTS.md
- review/context-qsa-closure-20261002/ROOT-FINAL-REVIEW.json
- review/context-qsa-run4-postprocess-20261003/PER-REQUEST-CROSS-REVIEW.json
必要时下钻原始 SSE、冻结 plan、planner CUDA、服务挂接与原 reference。不要只复述现有总结。

已知事实（均请核验）：
本次 context-qsa-run4 完成 19 行、两次重复、A0/B/A2 共114组、798 HTTP。32K/64K 的 c4 客户端共同解码指标约改善0.58%–0.78%；部分 c8 场景变号。64K/c8 六组都没有共同解码窗口，必须保留缺失。146个主请求三元组：88全相等、48基线自身变化、10个A0=A2但B不同；整组分类会掩盖后一类。生产恢复53请求/17门通过。n=2无CI，SSE不是GPU时间，静态候选挂接也不是逐组命中证明。

重点回答：
1. 测量、指标、源码与对齐方式是否支持当前结论？是否有分析漏洞、混杂因素或遗漏的反例？同长度QSA实验与历史mixed/staggered调度问题分开评估。
2. 10处输出分歧能证明什么、不能证明什么？不要把它们全部归于QSA，也不要以基线自变为由忽略；若需定位，最小可证伪检查是什么？
3. 小幅收益是真实信号还是尚不能与漂移/路径差异区分？结合可部署收益、复杂度和验证成本，独立建议封存、只做一次有限定位，还是继续开发。不要预设“小于1%必然没价值”，也不要为了继续优化而放宽门槛。
4. 若目标仍是接近物理极限，下一步最值得测的瓶颈是什么？已有证据、假设、所需profile和预期信息增益分别说明，不臆造剩余加速空间。

只审查现有证据；不要重跑GPU benchmark、重启/修改服务、安装依赖、改候选/冻结结果、push或合并。若做CPU复算，使用主机既有 /home/l/work/1Cat-vLLM/.venv/bin/python 或项目声明的uv环境，不使用系统Python。旧归档中的6条CPU夹具链接在Linux上悬空，不能假定所有本机脚本可直接在主机重放。

请在 D/review/ 下新建独立review目录保存 REVIEW.md（只新增报告，不覆盖已有材料），输出：按重要性排序的发现及文件/行号或具体请求证据；哪些结论成立/不成立/仍未知；明确的投入建议；如建议追加验证，给出最小范围、成本上限、通过/停止条件。先交review意见，不直接执行新实验。

"""Render a complete matrix for human review without making admission decisions."""
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
SOURCE = BASE / 'context-qsa-run4/analysis/RESULT.json'
PLAN = BASE / 'context-qsa-run4/control/plan.json'
PLAN_SHA = '355ef8caeee318fc79abad12cf4dcf70b05dd3a822177eb17b03ec2675b64ba4'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numbers(values):
    return ' / '.join('缺失' if v is None else f'{v:.3f}' for v in values)


def main():
    assert digest(PLAN) == PLAN_SHA
    value = json.loads(SOURCE.read_text())
    assert value['HTTP_requests'] == 798 and value['outputs'] == 112488
    assert value['n_per_row'] == 2 and len(value['rows']) == 19
    lines = [
        '# 跨上下文 A/B/A 结果', '',
        '每行固定两次配对，仅描述本轮样本。正收益表示 B 的客户端每 token 间隔较两份 A 的均值更短；无置信区间或性能准入。', '',
        f'原始分析状态：`{value["status"]}`；分析退出码：`{value["diagnostic_exit"]}`。', '',
        '| 输入 tokens | 并发 | A0 / B / A2 中位间隔 ms/token | 两次配对收益 % | 两次 A/A 漂移 % | 原整组输出分类 |',
        '|---:|---:|---|---|---|---|',
    ]
    for row in value['rows']:
        metric = row['common_decode']
        triples = metric['paired_values']
        assert len(triples) == 2 and all(len(t) == 3 for t in triples)
        medians = [None if any(t[i] is None for t in triples)
                   else statistics.median(t[i] for t in triples) for i in range(3)]
        tokens = sorted({r['status'] for r in row['token_equality']})
        lines.append(f'| {row["input_tokens"]} | {row["concurrency"]} | '
                     f'{numbers(medians)} | '
                     f'{numbers(metric.get("improvement_pct_per_pair", [None, None]))} | '
                     f'{numbers(metric.get("AA_drift_pct_per_pair", [None, None]))} | '
                     f'{"; ".join(tokens)} |')
    lines += ['', '## B 分别对两个 A 的配对收益', '',
        '每格按 repeat 0 / repeat 1 排列。正值代表 B 更快；保留全部行和缺失值。', '',
        '| 输入 tokens | 并发 | B 对 A0 % | B 对 A2 % |',
        '|---:|---:|---|---|']
    for row in value['rows']:
        triples = row['common_decode']['paired_values']
        b0 = [None if a0 is None or b is None else 100 * (1 - b / a0)
              for a0, b, a2 in triples]
        b2 = [None if a2 is None or b is None else 100 * (1 - b / a2)
              for a0, b, a2 in triples]
        lines.append(f'| {row["input_tokens"]} | {row["concurrency"]} | '
                     f'{numbers(b0)} | {numbers(b2)} |')
    lines += ['',
        '原整组分类不可用于排除组内个别请求的 B 分歧。逐请求 raw SSE 补充复核见 ROOT-FINAL-REVIEW.json：146 个请求三元组中，88 全相等、48 为 A/A 自身差异、10 为 A0=A2 但 B 不同。原分析及其 exit=3 保留不改；这些计数不构成 QSA 因果归因。', '',
        '各请求 TTFT、冷 prime TTFT 和完整请求耗时在原始 analysis/RESULT.json 中分别保留。输出 token 相等不代表这些长度的 logits 已逐位验证。', '',
        'c1–3 为回退路径，近256K的c1/c2仅验证兼容性。c4–8有静态候选挂接，未逐组记录动态命中。>32K为周期合成输入。GPU采样边界不是连续峰值；客户端SSE时间不是GPU执行时间或物理下界。', '',
        f'输入分析 SHA256：`{digest(SOURCE)}`。',
        f'计划 SHA256：`{PLAN_SHA}`。', '',
        '独立原始数据审计、显存审计和生产恢复结果由 REVIEW-PACK.md 分别链接。本表不覆盖旧 NO_GO。', '']
    with (HERE / 'CONTEXT-RESULTS.md').open('x') as stream:
        stream.write('\n'.join(lines))
    print('Rendered 19 descriptive rows; no performance admission inferred.')


if __name__ == '__main__':
    main()

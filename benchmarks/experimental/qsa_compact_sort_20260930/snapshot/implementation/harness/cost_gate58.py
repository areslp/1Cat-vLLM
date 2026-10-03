"""Planner pilot gate. A pass is permission for broader checks, not admission."""

import argparse
import json
from pathlib import Path
import statistics


parser = argparse.ArgumentParser()
parser.add_argument("--result", action="append", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
rows = []
for path in args.result:
    result = json.loads(Path(path).read_text())
    assert result["status"] == "PASS" and result["mode"] == "pilot"
    assert result["independent_capture_files"] == 33
    assert len(result["synthetic"]) == 9
    transition = result["metadata_graph_transition"]
    assert transition["one_graph"]
    assert transition["fresh_all_output_sentinels_each_replay"]
    assert [row["expected_hash_entries"] for row in transition["cases"]] == (
        [0, 1, 1024, 1025, 2048, 2049, 4096, 4097, 8192, 0, 1])
    assert all(row["full_metadata_exact"] for row in transition["cases"])
    c8 = [row for row in result["timing"]
          if row["tag"] == "c8" and row["kind"] == "verify"]
    assert len(c8) == 24
    savings = [row["summary"]["graph"]["paired_saving_us"] for row in c8]
    lows = [row["summary"]["graph"]["paired_saving_min_us"] for row in c8]
    regressions = [row["label"] for row in result["timing"]
                   if row["summary"]["graph"]["candidate_us"] >
                   row["summary"]["graph"]["reference_us"] * 1.01]
    gain = statistics.mean(savings) * 12 / 1000
    low = statistics.mean(lows) * 12 / 1000
    rows.append({"gpu": result["gpu"], "result_path": path,
                 "c8_capture_calls": len(c8),
                 "weighted_saving12calls_ms": gain,
                 "paired_low12calls_ms": low,
                 "other_shape_regressions": regressions,
                 "pass": gain >= .25 and low > 0 and not regressions})
assert len({row["gpu"] for row in rows}) == len(rows)
assert {row["gpu"] for row in rows} in ({0}, {0, 1, 2, 3})
passed = all(row["pass"] for row in rows)
output = {
    "status": "PILOT_COST_PASS_NOT_ADMISSION" if passed else "NO-GO",
    "engineering_gate_pass": passed, "gpus": rows,
    "physical_gpus_complete": len(rows) == 4,
    "calls_per_target_step": 12,
    "weighting": "Equal real c8 capture call weight; GPU repeats are not "
                 "independent model samples; planner savings are not measured "
                 "service savings.",
    "full_forward_graph_W1": "PENDING_NOT_EXECUTED",
    "next": "review broader correctness after all-four GPU cost gate" if
            passed else "stop before full admission/service shadow/A-B-A"}
Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output))
raise SystemExit(0 if passed else 10)

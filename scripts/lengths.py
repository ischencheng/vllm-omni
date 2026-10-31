"""Audio length and WER per prompt, pooled over rounds and containers.

python lengths.py <run dir> [<run dir> ...]
"""

import collections
import json
import math
import statistics
import sys
from pathlib import Path

rows = collections.defaultdict(lambda: collections.defaultdict(list))
for run in sys.argv[1:]:
    for res_file in sorted(Path(run).glob("*-r*/results.json")):
        res = json.loads(res_file.read_text())
        for key, val in res.items():
            if not key.startswith("c") or not key[1:].isdigit():
                continue
            for r in val["rows"]:
                if r.get("audio_s"):
                    rows[res["arm"]][r["prompt"]].append((r["audio_s"], r.get("wer"), r.get("ttfa_s")))
prompts = sorted(set(rows["off"]) & set(rows["dspark"]))
diffs, longer = [], 0
print(f"{'prompt':40s} {'n':>4s} {'off s':>7s} {'dspark s':>9s} {'diff %':>7s}")
for p in prompts:
    a = [x[0] for x in rows["off"][p]]
    b = [x[0] for x in rows["dspark"][p]]
    d = (statistics.mean(b) / statistics.mean(a) - 1) * 100
    diffs.append(d)
    longer += d > 0
    print(f"{p[:40]:40s} {len(a):4d} {statistics.mean(a):7.2f} {statistics.mean(b):9.2f} {d:7.1f}")
for arm in ("off", "dspark"):
    allv = [x for p in prompts for x in rows[arm][p]]
    secs = [x[0] for x in allv]
    wers = [x[1] for x in allv if x[1] is not None]
    print(arm, "n", len(secs), "mean audio s %.2f" % statistics.mean(secs), "sd %.2f" % statistics.pstdev(secs),
          "WER median %.3f mean %.3f" % (statistics.median(wers), statistics.mean(wers)),
          "WER>0.2: %d" % sum(w > 0.2 for w in wers))
m, s = statistics.mean(diffs), statistics.stdev(diffs)
print("per-prompt mean diff %.2f%%, sd %.2f, t=%.2f over %d prompts; dspark longer in %d" % (m, s, m / (s / math.sqrt(len(diffs))), len(diffs), longer))

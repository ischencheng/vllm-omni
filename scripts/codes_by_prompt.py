"""Greedy Talker codes per prompt and phase, matched by chat completion id.

python codes_by_prompt.py <run dir>  ->  agreement table between arms and phases.
"""

import collections
import json
import sys
from pathlib import Path


def load(arm_dir):
    codes = collections.defaultdict(list)
    for line in (arm_dir / "codes.jsonl").read_text().splitlines():
        rec = json.loads(line)
        codes[rec["req"].rsplit("-", 1)[0]] += rec["codes"]
    res = json.loads((arm_dir / "results.json").read_text())
    out = {}
    for phase in ("c1", "c16"):
        for i, row in enumerate(res.get(phase, {}).get("rows", [])):
            ids = row.get("ids") or []
            out[(phase, i)] = (row["prompt"], codes.get(ids[0]) if ids else None)
    return out


def first_diff(a, b):
    j = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return j if j is not None or len(a) == len(b) else min(len(a), len(b))


run = Path(sys.argv[1])
arms = {d.name: load(d) for d in sorted(run.glob("*-r*")) if (d / "codes.jsonl").exists()}
pairs = [("off-r0", "c1", "off-r0", "c16"), ("dspark-r0", "c1", "dspark-r0", "c16"),
         ("off-r0", "c1", "dspark-r0", "c1"), ("off-r0", "c16", "dspark-r0", "c16")]
for a_arm, a_ph, b_arm, b_ph in pairs:
    if a_arm not in arms or b_arm not in arms:
        continue
    a = {p: c for (ph, _), (p, c) in arms[a_arm].items() if ph == a_ph}
    b = collections.defaultdict(list)
    for (ph, _), (p, c) in arms[b_arm].items():
        if ph == b_ph:
            b[p].append(c)
    same = total = 0
    splits = []
    for prompt, codes in a.items():
        for other in b.get(prompt, []):
            if codes is None or other is None:
                continue
            total += 1
            d = first_diff(codes, other)
            same += d is None
            if d is not None:
                splits.append(d)
    print(f"{a_arm} {a_ph} vs {b_arm} {b_ph}: identical {same}/{total}, splits at {sorted(splits)}")

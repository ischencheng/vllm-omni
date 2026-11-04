"""Reply tokens vs rows per dumped handoff: python rows_gap.py <arm dir>. Only the last token may lack a row."""
import sys
from pathlib import Path

import torch

for arm in map(Path, sys.argv[1:]):
    gaps = {}
    for f in (arm / "rows").glob("*.pt"):
        d = torch.load(f, weights_only=False)
        g = len(d["ids"]) - d["rows"].shape[0]
        gaps[g] = gaps.get(g, 0) + 1
    print(arm.name, "reply tokens minus rows:", dict(sorted(gaps.items())))

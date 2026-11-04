"""Thinker rows the Talker got, first pass (miss) vs second pass (hit), one server.

python compare_rows.py <arm dir with rows/*.pt>   (needs torch)
"""

import sys
from collections import defaultdict
from pathlib import Path

import torch


def main():
    for arm in map(Path, sys.argv[1:]):
        by_prompt = defaultdict(list)
        for f in (arm / "rows").glob("*.pt"):
            d = torch.load(f, weights_only=False)
            by_prompt[d["prompt_hash"]].append(d)
        print(f"== {arm.name}: {len(by_prompt)} prompts")
        for items in by_prompt.values():
            if len(items) < 2:
                continue
            items.sort(key=lambda d: d["time"])
            a, b = items[0], items[1]
            n = 0
            for x, y in zip(a["ids"], b["ids"]):
                if x != y:
                    break
                n += 1
            n = min(n, a["rows"].shape[0], b["rows"].shape[0])
            ra, rb = a["rows"][:n].float(), b["rows"][:n].float()
            cos = torch.nn.functional.cosine_similarity(ra, rb, dim=-1)
            shift = torch.nn.functional.cosine_similarity(ra[:-1], rb[1:], dim=-1) if n > 1 else torch.tensor([float("nan")])
            print(f"  prompt_len={a['prompt_len']} ids {len(a['ids'])}/{len(b['ids'])} rows {a['rows'].shape[0]}/{b['rows'].shape[0]} "
                  f"shared={n} cos_min={cos.min().item():.4f} max_abs={(ra - rb).abs().max().item():.3f} shift_mean={shift.mean().item():.3f}")


if __name__ == "__main__":
    main()

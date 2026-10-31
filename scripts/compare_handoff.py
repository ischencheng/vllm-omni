"""Compare the Talker handoff (token ids + Thinker rows) between the off and DSpark arms.

python compare_handoff.py <run dir>   (a smoke.py --audio run with SPEC_DUMP_DIR=<run>/dumps)

For each prompt, match each arm's dump by response id, take the longest common
token prefix of the two handoffs, and compare the Thinker rows on it row by row.
Aligned rows agree up to numerical noise; a row shifted by one token would not.
Also reports the row-vs-token count of every handoff.
"""

import json
import sys
from pathlib import Path

import torch


def load_arm(run: Path, arm: str):
    res = json.loads((run / f"{arm}-r0" / "results.json").read_text())
    dumps = {p.stem: torch.load(p, weights_only=False) for p in (run / "dumps").glob("*.pt")}
    out = []
    for row in res["c1"]["rows"]:
        rid = row.get("id") or ""
        match = [d for name, d in dumps.items() if rid and name.startswith(rid)]
        out.append(match[0] if match else None)
    return res["c1"]["rows"], out


def main():
    run = Path(sys.argv[1])
    off_rows, off = load_arm(run, "off")
    ds_rows, ds = load_arm(run, "dspark")
    report = []
    for i, (a, b) in enumerate(zip(off, ds)):
        if a is None or b is None:
            report.append({"idx": i, "missing": [a is None, b is None]})
            continue
        ia, ib = a["ids"].tolist(), b["ids"].tolist()
        n = 0
        while n < min(len(ia), len(ib)) and ia[n] == ib[n]:
            n += 1
        ha, hb = a["hidden"][:n].float(), b["hidden"][:n].float()
        cos = torch.nn.functional.cosine_similarity(ha, hb, dim=-1) if n else torch.empty(0)
        # Same comparison with B shifted by one row: what a misaligned ledger would give.
        shifted = (
            torch.nn.functional.cosine_similarity(ha[1:], hb[:-1], dim=-1) if n > 1 else torch.empty(0)
        )
        report.append({
            "idx": i,
            "same_text": off_rows[i].get("text") == ds_rows[i].get("text"),
            "off_ids": len(ia), "off_rows": int(a["hidden"].shape[0]),
            "ds_ids": len(ib), "ds_rows": int(b["hidden"].shape[0]),
            "common_prefix": n,
            "cos_min": round(float(cos.min()), 5) if n else None,
            "cos_mean": round(float(cos.mean()), 5) if n else None,
            "cos_shifted_mean": round(float(shifted.mean()), 5) if n > 1 else None,
            "max_abs_diff": round(float((ha - hb).abs().max()), 4) if n else None,
            "wer_off": off_rows[i].get("asr_wer_vs_text"), "wer_ds": ds_rows[i].get("asr_wer_vs_text"),
            "wer_chunked_off": off_rows[i].get("asr_chunked_wer_vs_text"),
            "wer_chunked_ds": ds_rows[i].get("asr_chunked_wer_vs_text"),
        })
    print(json.dumps(report, indent=1))
    (run / "handoff_compare.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

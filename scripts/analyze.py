"""Summarize pc_bench runs: python analyze.py <run dir> [...]  (local, no GPU deps)."""

import json
import re
import statistics as st
import sys
from pathlib import Path

WORKLOADS = ["correct_pass1", "correct_pass2", "correct_mm_pass1", "correct_mm_pass2", "edge", "stress", "miss_c1", "sys_c1", "sys_c8", "long_c1", "chat_c4", "text_short_miss", "text_short_hit", "text_long_miss", "text_long_hit", "miss_c8"]


def pct(vals, q):
    vals = sorted(vals)
    return vals[int(q * (len(vals) - 1))] if vals else float("nan")


def metric_totals(path):
    out = {}
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        m = re.match(r"vllm:prefix_cache_(hits|queries)_total\{.*stage=\"(\d)\"\} ([0-9.e+]+)", ln)
        if m and m.group(2) == "0":
            out[m.group(1)] = float(m.group(3))
    return out


def main():
    for run in map(Path, sys.argv[1:]):
        for arm_dir in sorted(p for p in run.iterdir() if p.is_dir() and re.match(r"v\d-", p.name)):
            print(f"== {run.name}/{arm_dir.name}")
            prev = {}
            for w in WORKLOADS:
                f = arm_dir / f"{w}.json"
                if not f.exists():
                    continue
                d = json.loads(f.read_text())
                rows = d["rows"]
                ok = [r for r in rows if r.get("status") == 200 and not r.get("error")]
                if w in ("sys_c1",):
                    ok_t = ok[1:]  # the first one seeds the cache
                else:
                    ok_t = ok
                ttft = [r["ttft_s"] * 1000 for r in ok_t if "ttft_s" in r]
                ttfa = [r["ttfa_s"] * 1000 for r in ok_t if "ttfa_s" in r]
                lat = [r["latency_s"] for r in ok_t]
                audio = sum(1 for r in ok if r.get("audio_s"))
                wers = [r["wer"] for r in ok if "wer" in r]
                ptok = [r.get("usage", {}).get("prompt_tokens") for r in ok if r.get("usage")]
                cached = [(r.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens") for r in ok if r.get("usage")]
                extra = f" wer_med={st.median(wers):.3f}" if wers else ""
                print(f"  {w:14s} n={len(rows):3d} ok={len(ok):3d} audio={audio:3d} wall={d['wall_s']:7.1f}s "
                      f"ttft p50/p90={pct(ttft,.5):6.0f}/{pct(ttft,.9):6.0f} ttfa p50/p90={pct(ttfa,.5):6.0f}/{pct(ttfa,.9):6.0f} "
                      f"lat_mean={st.mean(lat) if lat else float('nan'):5.2f} prompt~{ptok[len(ptok)//2] if ptok else '-'} "
                      f"cached~{cached[len(cached)//2] if cached else '-'}{extra} state={d.get('state',{}).get('host_rss_tree_gb')}G/{d.get('state',{}).get('gpu_mem_mib_util')}")
                m = metric_totals(arm_dir / f"{w}.metrics.txt")
                if m:
                    dq, dh = m["queries"] - prev.get("queries", 0), m["hits"] - prev.get("hits", 0)
                    print(f"    stage0 prefix cache this workload: queries {dq:.0f} hits {dh:.0f} ({100 * dh / max(dq, 1):.0f}%)")
                    prev = m
            if (arm_dir / "correct_pass1.json").exists() and (arm_dir / "correct_pass2.json").exists():
                a = json.loads((arm_dir / "correct_pass1.json").read_text())["rows"]
                b = json.loads((arm_dir / "correct_pass2.json").read_text())["rows"]
                same = sum(x.get("text") == y.get("text") for x, y in zip(a, b))
                print(f"  text pass1==pass2: {same}/{len(a)}  ttft p50 {pct([r.get('ttft_s',0)*1000 for r in a],.5):.0f}->{pct([r.get('ttft_s',0)*1000 for r in b],.5):.0f}"
                      f"  ttfa p50 {pct([r.get('ttfa_s',0)*1000 for r in a],.5):.0f}->{pct([r.get('ttfa_s',0)*1000 for r in b],.5):.0f}")


if __name__ == "__main__":
    main()

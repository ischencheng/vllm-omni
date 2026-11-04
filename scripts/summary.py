"""python summary.py <run> [<run> ...]: per workload and arm, TTFT p50 / TTFA p50 / TTFA p90 / wall."""
import json
import statistics as st
import sys
from pathlib import Path

ARMS = ["v1-off", "v1-on", "v2-off", "v2-merge", "v2-native", "v2-off-ds", "v2-native-ds", "v2-off-fp32", "v2-native-fp32", "v2-native-tiny", "v2-off-tiny"]
WL = ["stress", "correct_mm_pass1", "correct_mm_pass2", "miss_c1", "sys_c1", "sys_c8", "long_c1", "chat_c4", "miss_c8", "text_short_miss", "text_short_hit", "text_long_miss", "text_long_hit"]
runs = [Path(a) for a in sys.argv[1:]]


def pct(v, q):
    v = sorted(v)
    return v[int(q * (len(v) - 1))] if v else float("nan")


print("TTFT p50, TTFA p50, TTFA p90, wall (one column per container)")
for w in WL:
    print("--", w)
    for arm in ARMS:
        out = []
        for run in runs:
            f = run / f"{arm}-r0" / f"{w}.json"
            if not f.exists():
                out.append("-")
                continue
            d = json.loads(f.read_text())
            rows = d["rows"]
            ok = [r for r in rows if r.get("status") == 200 and not r.get("error")]
            bad, noa = len(rows) - len(ok), sum(1 for r in ok if not r.get("audio_s"))
            if w == "sys_c1":
                ok = ok[1:]
            ttft = [r["ttft_s"] * 1e3 for r in ok if "ttft_s" in r]
            ttfa = [r["ttfa_s"] * 1e3 for r in ok if "ttfa_s" in r]
            out.append(f"{pct(ttft, .5):5.0f} {pct(ttfa, .5):5.0f} {pct(ttfa, .9):5.0f} {d['wall_s']:5.1f}s" + (f" bad={bad}" if bad else "") + (f" noaudio={noa}" if noa and ttfa else ""))
        if any(o != "-" for o in out):
            print(f"  {arm:13s}", " | ".join(out))
print()
for run in runs:
    for arm in ARMS:
        a = run / f"{arm}-r0"
        if not (a / "correct_pass1.json").exists():
            continue
        p1 = json.loads((a / "correct_pass1.json").read_text())["rows"]
        p2 = json.loads((a / "correct_pass2.json").read_text())["rows"]
        same = sum(x.get("text") == y.get("text") for x, y in zip(p1, p2))
        w1 = [r.get("wer") for r in p1 if "wer" in r] or [float("nan")]
        w2 = [r.get("wer") for r in p2 if "wer" in r] or [float("nan")]
        last = json.loads((a / "miss_c8.json").read_text())["state"] if (a / "miss_c8.json").exists() else {}
        print(run.name[:2], f"{arm:13s} text pass1==pass2 {same}/{len(p1)} wer med {st.median(w1):.3f}/{st.median(w2):.3f} max {max(w1):.2f}/{max(w2):.2f} "
              f"host_used {last.get('host_used_gb')} rss {last.get('host_rss_tree_gb')} gpu {last.get('gpu_mem_mib_util')}")

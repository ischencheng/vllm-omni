"""Summarize a spec_pc.py run.

uv run --no-project --with torch python control/analyze.py <run dir> [--round 1] [--ref main-on]

* Per arm: text-only TTFT and decode rate (completion tokens after the first,
  over the span from first to last text chunk), tokens committed per verify
  step, TTFA and audio length, split by miss (turn 0) and hit (later turns).
* Per turn: tokens per verify for the draft arms side by side (same prompts).
* Text agreement with --ref per request, and the Talker rows each arm handed
  over (rows/*.pt, sorted by time; the audio requests run one at a time):
  ids equal, and the lowest row cosine over the common ids.
"""

import argparse
import json
import statistics
from pathlib import Path


def load(run, rnd):
    arms = {}
    for d in sorted(Path(run).glob(f"*-r{rnd}")):
        f = d / "results.json"
        if f.exists():
            arms[d.name.rsplit("-r", 1)[0]] = (d, json.loads(f.read_text()))
    return arms


def ok(rows):
    return [r for r in rows if r.get("status") == 200 and not r.get("error")]


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def fmt(v, scale=1.0, nd=1):
    return "-" if v is None else f"{v * scale:.{nd}f}"


def rate(r):
    if r.get("text_span_s") and (r.get("completion_tokens") or 0) > 1:
        return (r["completion_tokens"] - 1) / r["text_span_s"]
    return None


def tpv(rows):
    steps = sum(r["verify_steps"] for r in rows)
    return sum(r["verify_tokens"] for r in rows) / steps if steps else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--ref", default="main-on")
    args = ap.parse_args()
    arms = load(args.run, args.round)
    print(f"round {args.round}: {', '.join(arms)}")
    print("arm            | text TTFT ms miss/hit | decode tok/s miss/hit | tok/verify miss/hit | TTFA ms miss/hit | audio s | cached tok hit | errors")
    for name, (_, res) in arms.items():
        if not res.get("ready"):
            print(f"{name:14s} | not ready: {(res.get('startup_errors') or ['?'])[0][-200:]}")
            continue
        rows = res.get("rows", [])
        text = ok([r for r in rows if r["workload"] == "text"])
        audio = ok([r for r in rows if r["workload"] == "conv_audio"])
        tm, th = [r for r in text if r["turn"] == 0], [r for r in text if r["turn"] > 0]
        am, ah = [r for r in audio if r["turn"] == 0], [r for r in audio if r["turn"] > 0]
        errs = sum(1 for r in rows if r.get("error") or r.get("status") != 200)
        noaudio = sum(1 for r in audio if not r.get("audio_s"))
        print(f"{name:14s} | {fmt(mean(r.get('ttft_s') for r in tm), 1e3, 0):>5s} / {fmt(mean(r.get('ttft_s') for r in th), 1e3, 0):>5s}     "
              f"| {fmt(mean(rate(r) for r in tm)):>6s} / {fmt(mean(rate(r) for r in th)):>6s}      "
              f"| {fmt(tpv(tm + am), 1, 2):>5s} / {fmt(tpv(th + ah), 1, 2):>5s}       "
              f"| {fmt(mean(r.get('ttfa_s') for r in am), 1e3, 0):>5s} / {fmt(mean(r.get('ttfa_s') for r in ah), 1e3, 0):>5s} "
              f"| {fmt(mean(r.get('audio_s') for r in audio), 1, 2)} | {fmt(mean(r.get('cached_tokens') for r in th + ah), 1, 0)} "
              f"| {errs} err, {noaudio} no audio")

    draft = [n for n, (_, res) in arms.items() if res.get("draft") == "ds" and res.get("ready")]
    if draft:
        print("\ntokens per verify by turn (text + audio workloads):")
        print("turn | " + " | ".join(f"{n:>12s}" for n in draft))
        for t in range(8):
            cells = []
            for n in draft:
                rows = [r for r in ok(arms[n][1]["rows"]) if r["turn"] == t and r["workload"] in ("text", "conv_audio")]
                cells.append(f"{fmt(tpv(rows), 1, 2):>12s}")
            if any(c.strip() != "-" for c in cells):
                print(f"{t:4d} | " + " | ".join(cells))

    if args.ref in arms and arms[args.ref][1].get("ready"):
        ref_rows = arms[args.ref][1]["rows"]
        print(f"\ntext equal to {args.ref} (text / conv_audio workloads, miss | hit):")
        for name, (_, res) in arms.items():
            if name == args.ref or not res.get("ready"):
                continue
            out = []
            for wl in ("text", "conv_audio"):
                pairs = [(a, b) for a, b in zip(ref_rows, res["rows"]) if a["workload"] == wl == b["workload"]]
                for label, sel in (("miss", [p for p in pairs if p[0]["turn"] == 0]), ("hit", [p for p in pairs if p[0]["turn"] > 0])):
                    out.append(f"{wl}-{label} {sum(a['text'] == b['text'] for a, b in sel)}/{len(sel)}")
            print(f"  {name}: " + ", ".join(out))
        try:
            import torch
        except ImportError:
            return
        def dumps(d):
            items = [torch.load(p, weights_only=False) for p in (d / "rows").glob("*.pt")]
            return sorted(items, key=lambda x: x["time"])
        ref = dumps(arms[args.ref][0])
        print(f"\nTalker rows vs {args.ref} ({len(ref)} dumps):")
        for name, (d, res) in arms.items():
            if name == args.ref or not res.get("ready"):
                continue
            other = dumps(d)
            same_ids, worst, n = 0, 1.0, min(len(ref), len(other))
            for a, b in zip(ref, other):
                k = 0
                while k < min(len(a["ids"]), len(b["ids"])) and a["ids"][k] == b["ids"][k]:
                    k += 1
                same_ids += a["ids"] == b["ids"]
                k = min(k, a["rows"].shape[0], b["rows"].shape[0])
                if k:
                    cos = torch.nn.functional.cosine_similarity(a["rows"][:k].float(), b["rows"][:k].float(), dim=-1)
                    worst = min(worst, float(cos.min()))
            print(f"  {name}: {len(other)} dumps, ids equal {same_ids}/{n}, min row cos over common ids {worst:.4f}")


if __name__ == "__main__":
    main()

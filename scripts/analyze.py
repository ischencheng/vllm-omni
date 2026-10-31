"""Summarize a talker_smoke.py run directory.

python analyze.py <run dir>

* per arm/round/concurrency: requests ok, wall time, req/s, audio s/s, TTFA and
  latency median/p90, Whisper WER median/mean;
* codes handed to Code2Wav (debug tree's codes.jsonl): per request, the same
  prompt in the off and dspark arms of a round -- identical, or the first
  differing code index;
* SPECACC lines (debug tree, SPEC_DEBUG=1) after the timed marker: codes
  committed per verify step for full verify steps (k drafts on every request),
  split by phase (c1, c16).
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def codes_by_request(path):
    """Codes per request id in arrival order, and the order requests first appear."""
    codes, order = defaultdict(list), []
    if not path.exists():
        return codes, order
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec["req"] not in codes:
            order.append(rec["req"])
        codes[rec["req"]].extend(rec["codes"])
    return codes, order


def specacc(server_log, k, marks):
    """marks: phase -> number of SPECACC lines written before the phase started."""
    bounds = sorted(marks.items(), key=lambda kv: kv[1])
    stats = defaultdict(lambda: [0, 0, defaultdict(int)])
    lines = [ln for ln in server_log.read_text(errors="replace").splitlines() if "SPECACC" in ln]
    for idx, line in enumerate(lines):
        phase = next((name for name, start in reversed(bounds) if idx >= start), None)
        m = re.search(r"SPECACC nreq=(\d+) ntok=(\d+) ndraft=(\d+) nsamp=\[([^\]]*)\]", line)
        if not m or phase is None:
            continue
        nreq, ntok, ndraft = map(int, m.groups()[:3])
        sampled = [int(x) for x in m.group(4).split(",") if x.strip()]
        if ndraft == k * nreq and ntok == (k + 1) * nreq:
            entry = stats[phase]
            entry[0] += len(sampled)
            entry[1] += sum(sampled)
            for s in sampled:
                entry[2][s] += 1
    return {p: {"request_steps": s, "codes_per_step": t / s if s else None, "hist": dict(sorted(h.items()))}
            for p, (s, t, h) in stats.items()}


def main():
    run = Path(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    report = {"runs": [], "codes": [], "acceptance": {}}
    per_arm_codes = {}
    for res_file in sorted(run.glob("*-r*/results.json")):
        res = json.loads(res_file.read_text())
        arm, rnd = res["arm"], res["round"]
        for key, val in res.items():
            if not re.fullmatch(r"c\d+", key):
                continue
            rows = val["rows"]
            ok = [r for r in rows if not r.get("error") and r.get("audio_s")]
            ttfa = [r["ttfa_s"] for r in ok if "ttfa_s" in r]
            lat = [r["latency_s"] for r in ok]
            wers = [r["wer"] for r in ok if "wer" in r]
            report["runs"].append({
                "arm": arm, "round": rnd, "conc": key, "ok": f"{len(ok)}/{len(rows)}",
                "wall_s": round(val["wall_s"], 2), "req_per_s": round(val["req_per_s"] or 0, 3),
                "audio_s_per_s": round(val["audio_s_per_s"] or 0, 2),
                "ttfa_p50": pct(ttfa, 0.5), "ttfa_p90": pct(ttfa, 0.9),
                "latency_p50": pct(lat, 0.5), "latency_p90": pct(lat, 0.9),
                "audio_s": round(sum(r["audio_s"] for r in ok), 1),
                "wer_median": statistics.median(wers) if wers else None,
                "wer_mean": statistics.mean(wers) if wers else None,
            })
        codes, order = codes_by_request(res_file.parent / "codes.jsonl")
        per_arm_codes[(arm, rnd)] = (codes, order, res)
        server_log = res_file.parent / "server.log"
        if server_log.exists() and arm != "off" and res.get("specacc_marks"):
            report["acceptance"][f"{arm}-r{rnd}"] = specacc(server_log, k, res["specacc_marks"])
    # c1 requests run one at a time after a 4-request warm-up: request i of the
    # c1 phase is the (4 + i)-th id to appear in the code dump.
    for rnd in sorted({r for (_, r) in per_arm_codes}):
        off, spec = per_arm_codes.get(("off", rnd)), per_arm_codes.get(("dspark", rnd))
        if not off or not spec:
            continue
        n = len(off[2].get("c1", {}).get("rows", []))
        for i in range(n):
            a_ids, b_ids = off[1][4:4 + n], spec[1][4:4 + n]
            if i >= len(a_ids) or i >= len(b_ids):
                break
            a, b = off[0][a_ids[i]], spec[0][b_ids[i]]
            first = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if first is None and len(a) != len(b):
                first = min(len(a), len(b))
            report["codes"].append({"round": rnd, "idx": i, "off_codes": len(a), "dspark_codes": len(b),
                                    "identical": first is None, "first_diff": first,
                                    "text_same": off[2]["c1"]["rows"][i].get("text") == spec[2]["c1"]["rows"][i].get("text")})
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

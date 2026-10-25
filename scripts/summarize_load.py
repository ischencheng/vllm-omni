"""Per-session frame latency and driver fields from a load run's load_<N>/load-result.json.

python summarize_load.py <run dir> [--out load-summary.json]

Frame latency of input frame k = received_at of the output packet that completes
output frame k+1, minus send_completed_at of input frame k, over frames 2..
(the snippet in the PR body). Memory is the max of gpu_mem.csv.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def frame_latency_ms(s):
    sent = [e["send_completed_at_s"] for e in s["input_timeline"]]
    got = [p["received_at_s"] for p in s["audio_packet_timeline"]]
    frames = np.cumsum([p["samples"] for p in s["audio_packet_timeline"]]) // 1920
    lat = np.array([got[np.searchsorted(frames, k + 1)] - sent[k] for k in range(2, min(len(sent), int(frames[-1])))])
    half = len(lat) // 2
    r = lambda x: round(float(x) * 1e3, 1)  # noqa: E731
    return {"median": r(np.median(lat)), "p99": r(np.percentile(lat, 99)), "max": r(lat.max()),
            "first_half_median": r(np.median(lat[:half])), "second_half_median": r(np.median(lat[half:]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    mem = None
    if (args.run_dir / "gpu_mem.csv").exists():
        vals = []
        for line in (args.run_dir / "gpu_mem.csv").read_text().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[1].replace(".", "").isdigit():
                vals.append(float(parts[1]))
        mem = max(vals) if vals else None
    loads = {}
    for d in sorted(args.run_dir.glob("load_*"), key=lambda p: int(p.name.split("_")[1]) if p.is_dir() else 0):
        f = d / "load-result.json"
        if not d.is_dir() or not f.exists():
            continue
        res = json.loads(f.read_text())
        loads[d.name.split("_")[1]] = {
            "ok": res.get("ok"), "passed_sessions": res.get("passed_sessions"),
            "reported_server": res.get("reported_server"),
            "sessions": [{
                "session": s.get("index"), "ok": s.get("ok"), "input_frames": s.get("input_frames"),
                "output_samples": s.get("output_samples"),
                "frame_latency_ms": frame_latency_ms(s),
                "client_send_lateness_ms": {k: round(s["client_send_lateness_ms"][k], 1) for k in ("median", "p99")},
                "client_pacing_warning": s.get("client_pacing_warning"),
                "client_stream_rtf": round(s.get("client_stream_rtf") or 0, 4),
                "audio_underrun_max_s": round(s.get("audio_underrun_s") or 0, 3),
                "audio_continuity_ok": s.get("audio_continuity_ok"),
                "first_audio_ms": round(s.get("client_first_audio_after_stream_start_ms") or 0, 1),
            } for s in res.get("sessions", [])],
        }
    out = {"gpu_memory_used_mib_max": mem, "loads": loads}
    text = json.dumps(out, indent=1)
    if args.out:
        args.out.write_text(text)
    for n, v in loads.items():
        lat = [s["frame_latency_ms"] for s in v["sessions"]]
        print(n, "median", min(x["median"] for x in lat), "-", max(x["median"] for x in lat),
              "p99", min(x["p99"] for x in lat), "-", max(x["p99"] for x in lat),
              "max", max(x["max"] for x in lat),
              "half-diff", max(abs(x["first_half_median"] - x["second_half_median"]) for x in lat),
              "lateness_p99", max(s["client_send_lateness_ms"]["p99"] for s in v["sessions"]),
              "continuity", sum(bool(s["audio_continuity_ok"]) for s in v["sessions"]), "/", len(v["sessions"]),
              "pacing_warn", sum(bool(s["client_pacing_warning"]) for s in v["sessions"]))
    print("gpu mem max MiB", mem)


if __name__ == "__main__":
    main()

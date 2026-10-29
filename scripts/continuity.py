"""Playback continuity of recorded duplex replies, with the repo's own helper.

python continuity.py <vllm-omni tree> <events_*.json> ...  ->  JSON lines on stdout

Loads `vllm_omni/benchmarks/audio_continuity.py` by path (it has no vllm
dependency) and feeds it each response's audio-delta arrival times (client
monotonic clock, `_client_received_at_s`) and PCM16 byte counts.
"""

import base64
import importlib.util
import json
import sys
from pathlib import Path


def load_helper(tree):
    path = Path(tree) / "vllm_omni" / "benchmarks" / "audio_continuity.py"
    spec = importlib.util.spec_from_file_location("audio_continuity", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audio_continuity"] = mod
    spec.loader.exec_module(mod)
    return mod.compute_continuity_stats


def main():
    compute = load_helper(sys.argv[1])
    for events_file in sys.argv[2:]:
        events = json.load(open(events_file))
        chunks = {}
        rates = {}
        for e in events:
            if e.get("type") != "response.output_audio.delta":
                continue
            rid = e.get("response_id") or (e.get("response") or {}).get("id")
            data = e.get("delta") or e.get("audio")
            if not data:
                continue
            chunks.setdefault(rid, []).append((float(e["_client_received_at_s"]), len(base64.b64decode(data))))
            if e.get("sample_rate_hz"):
                rates[rid] = int(e["sample_rate_hz"])
        for rid, rows in chunks.items():
            t = [r[0] for r in rows]
            n = [r[1] for r in rows]
            rate = rates.get(rid, 24000)
            stats = compute(t, n, rate)
            print(
                json.dumps(
                    {
                        "events": Path(events_file).name,
                        "response_id": rid,
                        "chunks": len(rows),
                        "audio_s": round(sum(n) / 2 / rate, 3),
                        "first_chunk_s": round(n[0] / 2 / rate, 3),
                        "last_arrival_after_first_s": round(t[-1] - t[0], 3),
                        "max_underrun_s": round(stats.max_underrun_s, 4),
                        "underrun_events": stats.underrun_event_count,
                        "continuous_100ms": stats.is_continuous,
                    }
                )
            )


if __name__ == "__main__":
    main()

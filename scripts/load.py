"""Moshi paced multi-session load with the repo's own driver (load mode), plus GPU memory.

Serves kyutai/moshiko-pytorch-bf16 on personaplex.yaml with max_sessions 16 (Stage 0
max_num_seqs 16, a cudagraph capture size; Stage 1 18), then for each N runs
  personaplex_realtime_duplex.py --sessions N --voice "" --persona "" ...
on the 30.7 s LibriSpeech input, sampling nvidia-smi memory.used every 0.5 s.
Outputs in $OUT_DIR: load_<N>/load-result.json, gpu_mem.csv, load.json.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parity  # noqa: E402
import probe  # noqa: E402

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
SRC = Path("/opt/vllm-omni")
REPO = parity.REPO


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--loads", default="1,4,8,16")
    ap.add_argument("--src-sha", default="unknown")
    args = ap.parse_args()
    loads = [int(x) for x in args.loads.split(",")]
    probe.sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv", log="gpu.txt")
    model_path = probe.step_download()
    inp = probe.step_input()
    deploy = probe.site_pkg() / "deploy" / "moshi_load.yaml"
    deploy.write_text(
        "base_config: personaplex.yaml\n"
        "duplex_session:\n  max_sessions: 16\n"
        "stages:\n"
        "  - stage_id: 0\n    max_num_seqs: 16\n"
        "  - stage_id: 1\n    max_num_seqs: 18\n"
    )
    smi = subprocess.Popen(
        "nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500",
        shell=True, stdout=open(OUT / "gpu_mem.csv", "w"), stderr=subprocess.STDOUT, start_new_session=True,
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    log = open(OUT / "server.log", "w")
    proc = subprocess.Popen(["vllm", "serve", REPO, "--omni", "--port", str(parity.PORT), "--deploy-config", str(deploy)],
                            stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    phases = {}
    try:
        t0 = time.time()
        import httpx

        while time.time() - t0 < 1200:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with {proc.returncode}")
            try:
                if httpx.get(f"http://127.0.0.1:{parity.PORT}/v1/models", timeout=5).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(5)
        phases["ready"] = time.time()
        print(f"ready after {phases['ready'] - t0:.0f}s", flush=True)
        for n in loads:
            t = time.time()
            r = probe.sh(
                f"{sys.executable} {SRC}/tests/e2e/online_serving/personaplex_realtime_duplex.py "
                f"--url ws://127.0.0.1:{parity.PORT}/v1/realtime?duplex=1 --model {REPO} --input-wav {inp} "
                f"--output-dir {OUT}/load_{n} --sessions {n} --voice '' --persona '' --secondary-persona '' "
                f"--replacement-persona '' --server-hardware '1x H100 80GB' --server-revision {args.src_sha}",
                timeout=1800, log=f"load_{n}.log")
            phases[f"load_{n}"] = (t, time.time(), r.returncode)
            res = json.loads((OUT / f"load_{n}" / "load-result.json").read_text()) if (OUT / f"load_{n}" / "load-result.json").exists() else {}
            sessions = res.get("sessions", [])
            summary = {
                "rc": r.returncode, "ok": res.get("ok"), "passed": res.get("passed_sessions"),
                "first_audio_ms": [s.get("client_first_audio_after_stream_start_ms") for s in sessions],
                "rtf": [s.get("client_stream_rtf") for s in sessions],
                "interval_p99_ms": [(s.get("client_audio_packet_interval_ms") or {}).get("p99") for s in sessions],
                "underrun_events": [s.get("audio_underrun_event_count") for s in sessions],
                "continuity_ok": [s.get("audio_continuity_ok") for s in sessions],
            }
            print(f"load {n}: {json.dumps(summary)}", flush=True)
            phases[f"summary_{n}"] = summary
            time.sleep(5)
    finally:
        parity.stop(proc, log)
        try:
            os.killpg(smi.pid, 15)
        except Exception:
            pass
    (OUT / "load.json").write_text(json.dumps(phases, indent=2, default=str))


if __name__ == "__main__":
    main()

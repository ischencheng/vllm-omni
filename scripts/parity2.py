"""Moshi on vLLM-Omni, second round: reference variability, repo-id serving, multi-session load, ASR.

Outputs in $OUT_DIR:
  ref_a / ref_b       two official greedy references (moshi.server framing), same input
  single              vLLM-Omni, deploy default (depformer CUDA graphs on), one session, local path
  repo                vLLM-Omni served by repo id instead of a local path, one 10 s session
  load_<n>            n concurrent paced sessions on max_sessions 8, per-session output timing
  asr.json            Whisper transcripts of the reference and vLLM-Omni outputs
  summary.json        agreement tables (ref vs ref, vLLM vs ref) and load metrics
"""

import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parity  # noqa: E402
import probe  # noqa: E402

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
CONTROL = Path(__file__).resolve().parent
REPO = parity.REPO


def run_ref(inp, tag):
    d = OUT / tag
    d.mkdir(exist_ok=True)
    r = probe.sh(f"{sys.executable} {CONTROL / 'ref_moshi.py'} --repo {REPO} --input {inp} --out-dir {d} "
                 "--semantics server", timeout=1800, log=f"{tag}.log")
    print(f"{tag}: rc={r.returncode}", flush=True)
    return dict(np.load(d / "ref_server_greedy.npz"))


def agreement(a_text, a_codes, b_text, b_codes):
    """Both in LMGen's de-delayed record layout ([text, cb0..cb7] per emitted step)."""
    n = min(len(a_text), len(b_text))
    t = a_text[:n] == b_text[:n]
    c = a_codes[:n] == b_codes[:n]
    nonpad = lambda x: [int(i) for i in np.nonzero((x != 0) & (x != 3))[0]]  # noqa: E731
    return {
        "steps": int(n),
        "text_match": float(t.mean()),
        "first_text_diff": int(np.argmin(t)) if not t.all() else int(n),
        "cb_match": [round(float(x), 3) for x in c.mean(axis=0)],
        "frame_match": float((t & c.all(axis=1)).mean()),
        "first_frame_diff": int(np.argmin(t & c.all(axis=1))) if not (t & c.all(axis=1)).all() else int(n),
        "a_nonpad_steps": nonpad(a_text[:n]),
        "b_nonpad_steps": nonpad(b_text[:n]),
    }


def vllm_records(text, codes):
    """Raw per-step samples -> LMGen's de-delayed records: cb0 of step j, cb1..7 of step j+1."""
    n = len(text) - 1
    return text[:n], np.concatenate([codes[:n, :1], codes[1 : n + 1, 1:8]], axis=1)


async def paced_sessions(model_path, pcm, n, tag):
    """n concurrent sessions, each streams pcm in real time; arrival time of every audio delta."""
    from vllm_omni.clients.duplex import AudioFormat, DuplexClient, SessionConfig

    config = SessionConfig(input_audio=AudioFormat("pcm_f32le", 24_000), output_audio=AudioFormat("pcm16", 24_000))
    url = f"ws://127.0.0.1:{parity.PORT}/v1/realtime?duplex=1"
    raw = pcm.astype("<f4").tobytes()

    async def one(i):
        arrivals, nbytes, errors = [], [], []
        t_start = None
        async with DuplexClient(url, model=model_path, config=config) as client:
            async def reader():
                async for event in client.events():
                    r = event.raw
                    if r.get("type") == "response.output_audio.delta":
                        delta = r.get("delta")
                        if isinstance(delta, str):
                            arrivals.append(time.time())
                            nbytes.append(len(base64.b64decode(delta)))
                    elif r.get("type") == "error":
                        errors.append(r)

            task = asyncio.create_task(reader())
            t_start = time.time()
            await client.stream_pcm(raw, chunk_ms=80, realtime=True)
            t_end_in = time.time()
            await asyncio.sleep(3.0)
            task.cancel()
        a = np.array(arrivals) - t_start
        samples = np.cumsum(np.array(nbytes) // 2) if nbytes else np.zeros(0)
        # Lateness of each delta vs. real time: the audio it completes should have been
        # playable at (start + its end sample / 24 kHz) + a fixed offset; report the spread.
        lag = a - samples / 24000.0 if len(a) else np.zeros(0)
        gaps = np.diff(a) if len(a) > 1 else np.zeros(0)
        return {
            "session": i, "deltas": len(a), "audio_s": float(samples[-1] / 24000.0) if len(samples) else 0.0,
            "input_s": t_end_in - t_start, "errors": len(errors),
            "lag_p50_ms": float(np.percentile(lag, 50) * 1e3) if len(lag) else None,
            "lag_p99_ms": float(np.percentile(lag, 99) * 1e3) if len(lag) else None,
            "lag_max_ms": float(lag.max() * 1e3) if len(lag) else None,
            "lag_growth_ms": float((np.median(lag[-50:]) - np.median(lag[:50])) * 1e3) if len(lag) > 100 else None,
            "gap_p99_ms": float(np.percentile(gaps, 99) * 1e3) if len(gaps) else None,
            "gaps_over_160ms": int((gaps > 0.16).sum()),
        }

    return await asyncio.gather(*(one(i) for i in range(n)))


def load_steps_window(trace_dir, t0, t1):
    rows = []
    for f in sorted(Path(trace_dir).glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("ev") == "post_sample" and t0 <= r["t"] <= t1:
                for i, rid in enumerate(r["req_ids"]):
                    rows.append((r["t"], rid, r["text"][i], r["codes"][i][:8]))
    rows.sort()
    rids = {rid for _, rid, _, _ in rows}
    if len(rids) != 1:
        print(f"single-session window holds {len(rids)} request ids", flush=True)
    return np.array([r[2] for r in rows]), np.array([r[3] for r in rows])


def step_stats(trace_dir, t0, t1):
    steps, posts = [], []
    for f in Path(trace_dir).glob("*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if not (t0 <= r.get("t", 0) <= t1):
                continue
            if r.get("ev") == "step" and r.get("m") == "execute_model" and r.get("nreq"):
                steps.append((r["t"], r["dur_ms"], r["nreq"]))
            elif r.get("ev") == "post_sample":
                posts.append(r["t"])
    if not steps:
        return {}
    d = np.array([s[1] for s in steps])
    nreq = np.array([s[2] for s in steps])
    return {"steps": len(steps), "exec_ms_p50": float(np.percentile(d, 50)), "exec_ms_p99": float(np.percentile(d, 99)),
            "nreq_mean": float(nreq.mean()), "nreq_max": int(nreq.max())}


def asr(paths):
    import soundfile as sf
    import soxr
    from transformers import pipeline

    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device=0)
    out = {}
    for tag, p in paths.items():
        if not Path(p).exists():
            continue
        a, sr = sf.read(p, dtype="float32")
        a16 = soxr.resample(a, sr, 16000).astype(np.float32)
        out[tag] = pipe({"raw": a16, "sampling_rate": 16000}, return_timestamps=True)["text"]
        print(f"asr {tag}: {out[tag]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loads", default="1,4,8")
    args = ap.parse_args()
    probe.sh("nvidia-smi --query-gpu=name,driver_version --format=csv", log="gpu.txt")
    model_path = probe.step_download()
    inp = probe.step_input()
    import soundfile as sf

    pcm, _ = sf.read(inp, dtype="float32")
    summary = {}
    ref_a, ref_b = run_ref(inp, "ref_a"), run_ref(inp, "ref_b")
    summary["ref_a_vs_ref_b"] = agreement(ref_a["text"], ref_a["codes"], ref_b["text"], ref_b["codes"])
    print("ref vs ref:", json.dumps(summary["ref_a_vs_ref_b"]), flush=True)

    # One session, deploy default (graphs on), local path.
    proc, log = parity.serve(model_path, "graphs", 8)
    try:
        t_single = time.time()
        summary["single_client"] = asyncio.run(parity.one_session(model_path, pcm, "single"))
        time.sleep(2)
        single_window = (t_single, time.time())
        loads = {}
        for n in [int(x) for x in args.loads.split(",")]:
            t0 = time.time()
            sessions = asyncio.run(paced_sessions(model_path, pcm, n, f"load_{n}"))
            t1 = time.time()
            loads[n] = {"sessions": sessions, "stage0": step_stats(OUT / "trace_graphs", t0, t1)}
            print(f"load {n}: {json.dumps(loads[n]['stage0'])} "
                  f"lag_p99={[round(s['lag_p99_ms'] or -1) for s in sessions]} "
                  f"gaps>160={[s['gaps_over_160ms'] for s in sessions]} errors={[s['errors'] for s in sessions]}",
                  flush=True)
            time.sleep(3)
        summary["load"] = loads
    finally:
        parity.stop(proc, log)
    text, codes = load_steps_window(OUT / "trace_graphs", *single_window)
    np.savez(OUT / "vllm_single.npz", text=text, codes=codes)
    vt, vc = vllm_records(text, codes)
    summary["vllm_vs_ref_a"] = agreement(ref_a["text"], ref_a["codes"], vt, vc)
    summary["vllm_vs_ref_b"] = agreement(ref_b["text"], ref_b["codes"], vt, vc)
    print("vllm vs ref_a:", json.dumps(summary["vllm_vs_ref_a"]), flush=True)

    # Repo id instead of a local snapshot path.
    try:
        proc, log = parity.serve(REPO, "repo", 2)
        try:
            summary["repo_client"] = asyncio.run(parity.one_session(REPO, pcm[: 24000 * 10], "repo"))
        finally:
            parity.stop(proc, log)
    except Exception as e:
        summary["repo_error"] = repr(e)
        print(f"repo arm failed: {e!r}", flush=True)

    summary["asr"] = asr({"ref_a": OUT / "ref_a" / "ref_server_greedy.wav", "ref_b": OUT / "ref_b" / "ref_server_greedy.wav",
                          "vllm_single": OUT / "vllm_single.wav", "input": inp})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

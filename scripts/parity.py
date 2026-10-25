"""Moshi on vLLM-Omni vs the official kyutai `moshi` package, greedy, one session.

Steps (outputs in $OUT_DIR):
  1. input wav (probe.step_input) and the official greedy reference in moshi.server
     framing (ref_moshi.py --semantics server)
  2. for each arm in --arms (eager: depformer CUDA graphs off; graphs: the deploy
     default), serve kyutai/moshiko-pytorch-bf16 on personaplex.yaml with the probe,
     stream the same wav as one realtime duplex session in 80 ms chunks, record
     every Stage 0 step's text token and raw depformer codes, save the audio
  3. compare each arm with the reference step by step (compare.json)
"""

import argparse
import asyncio
import base64
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe  # noqa: E402

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
CONTROL = Path(__file__).resolve().parent
REPO = "kyutai/moshiko-pytorch-bf16"
PORT = 8099


def serve(model_path, arm, max_sessions):
    deploy = probe.site_pkg() / "deploy" / f"moshi_{arm}.yaml"
    graphs = "true" if arm == "graphs" else "false"
    deploy.write_text(
        "base_config: personaplex.yaml\n"
        f"duplex_session:\n  max_sessions: {max_sessions}\n"
        "stages:\n"
        "  - stage_id: 0\n"
        "    hf_overrides:\n"
        "      mimi_cuda_graphs: true\n"
        f"      depformer_cuda_graphs: {graphs}\n"
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(CONTROL / "inject"),
           "MOSHI_TRACE_DIR": str(OUT / f"trace_{arm}")}
    log = open(OUT / f"server_{arm}.log", "w")
    cmd = ["vllm", "serve", model_path, "--omni", "--port", str(PORT), "--deploy-config", str(deploy)]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 1200:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/v1/models", timeout=5).status_code == 200:
                print(f"{arm}: ready after {time.time() - t0:.0f}s", flush=True)
                return proc, log
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("server not ready")


def stop(proc, log):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    subprocess.run("pkill -9 -f 'vllm serve' || true", shell=True)
    log.close()
    time.sleep(5)


async def one_session(model_path, pcm, tag, drain_s=3.0):
    from vllm_omni.clients.duplex import AudioFormat, DuplexClient, SessionConfig

    config = SessionConfig(input_audio=AudioFormat("pcm_f32le", 24_000), output_audio=AudioFormat("pcm16", 24_000))
    url = f"ws://127.0.0.1:{PORT}/v1/realtime?duplex=1"
    audio = bytearray()
    events = []
    t0 = time.time()
    async with DuplexClient(url, model=model_path, config=config) as client:
        async def reader():
            async for event in client.events():
                raw = event.raw if hasattr(event, "raw") else {}
                etype = raw.get("type")
                events.append({"t": time.time() - t0, "type": etype})
                if etype in ("response.output_audio.delta", "response.audio.delta"):
                    delta = raw.get("delta") or raw.get("audio")
                    if isinstance(delta, str):
                        audio.extend(base64.b64decode(delta))
                elif etype == "error":
                    print(f"{tag}: error event {raw}", flush=True)

        task = asyncio.create_task(reader())
        await client.stream_pcm(pcm.astype("<f4").tobytes(), chunk_ms=80, realtime=True)
        await asyncio.sleep(drain_s)
        task.cancel()
    pcm16 = np.frombuffer(bytes(audio), dtype="<i2")
    import soundfile as sf

    sf.write(OUT / f"vllm_{tag}.wav", pcm16.astype(np.float32) / 32768.0, 24000)
    types = {}
    for e in events:
        types[e["type"]] = types.get(e["type"], 0) + 1
    print(f"{tag}: {len(pcm16) / 24000:.1f}s audio out, events {types}", flush=True)
    return {"audio_s": len(pcm16) / 24000, "events": types}


def load_steps(trace_dir):
    rows = []
    for f in sorted(Path(trace_dir).glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("ev") == "post_sample":
                for i, rid in enumerate(r["req_ids"]):
                    rows.append((r["t"], rid, r["text"][i], r["codes"][i][:8]))
            elif r.get("ev") == "runtime":
                print("runtime:", r, flush=True)
    rows.sort()
    # One session: keep the request id that produced the most steps.
    counts = {}
    for _, rid, _, _ in rows:
        counts[rid] = counts.get(rid, 0) + 1
    if not counts:
        return np.zeros(0, int), np.zeros((0, 8), int)
    rid = max(counts, key=counts.get)
    sel = [r for r in rows if r[1] == rid]
    return np.array([r[2] for r in sel]), np.array([r[3] for r in sel])


def compare(ref, text, codes):
    """Reference record j = [text(step j), cb0(step j), cb1..7(step j+1)] (LMGen's de-delayed output)."""
    n = min(len(ref["text"]), len(text) - 1)
    rt, rc = ref["text"][:n], ref["codes"][:n]
    vt = text[:n]
    vc = np.concatenate([codes[:n, :1], codes[1 : n + 1, 1:8]], axis=1)
    text_eq = rt == vt
    frame_eq = text_eq & (rc == vc).all(axis=1)
    first_text_diff = int(np.argmin(text_eq)) if not text_eq.all() else n
    first_frame_diff = int(np.argmin(frame_eq)) if not frame_eq.all() else n
    return {
        "steps_compared": int(n),
        "text_match": float(text_eq.mean()),
        "frame_match": float(frame_eq.mean()),
        "cb_match": [float(x) for x in (rc == vc).mean(axis=0)],
        "first_text_diff": first_text_diff,
        "first_frame_diff": first_frame_diff,
        "ref_text_nonpad": int(((rt != 0) & (rt != 3)).sum()),
        "vllm_text_nonpad": int(((vt != 0) & (vt != 3)).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="eager,graphs")
    ap.add_argument("--max-sessions", type=int, default=2)
    args = ap.parse_args()
    probe.sh("nvidia-smi --query-gpu=name,driver_version --format=csv", log="gpu.txt")
    model_path = probe.step_download()
    inp = probe.step_input()
    r = probe.sh(f"{sys.executable} {CONTROL / 'ref_moshi.py'} --repo {REPO} --input {inp} --out-dir {OUT} "
                 "--semantics server", timeout=1800, log="ref_server_greedy.log")
    print(f"reference rc={r.returncode}", flush=True)
    ref = dict(np.load(OUT / "ref_server_greedy.npz"))
    import soundfile as sf

    pcm, _ = sf.read(inp, dtype="float32")
    tok = None
    try:
        import sentencepiece

        tok = sentencepiece.SentencePieceProcessor(str(Path(model_path) / "tokenizer_spm_32k_3.model"))
    except Exception:
        pass
    results = {"reference_text": json.loads((OUT / "ref_server_greedy.json").read_text())["text"]}
    for arm in args.arms.split(","):
        proc, log = serve(model_path, arm, args.max_sessions)
        try:
            results[f"client_{arm}"] = asyncio.run(one_session(model_path, pcm, arm))
            time.sleep(2)
        finally:
            stop(proc, log)
        text, codes = load_steps(OUT / f"trace_{arm}")
        np.savez(OUT / f"vllm_{arm}.npz", text=text, codes=codes)
        cmp = compare(ref, text, codes) if len(text) > 1 else {"steps_compared": 0}
        if tok is not None and len(text):
            cmp["text"] = "".join(tok.id_to_piece(int(t)).replace("▁", " ") for t in text if t not in (0, 3))
        results[arm] = cmp
        print(f"{arm}: {json.dumps({k: v for k, v in cmp.items() if k != 'cb_match'})}", flush=True)
    (OUT / "compare.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

"""Serve Moshi by Hub repo id with the Hub online (HF_HUB_OFFLINE unset), one 10 s session.

vLLM keeps `model_config.model` as the repo id when the Hub is online (it only
swaps in the snapshot path when HF_HUB_OFFLINE=1), so this is the case the
review on #9319 points at. Records whether the session opened as Moshi.

python repo_online.py  ->  $OUT_DIR/repo_online.json, server_repo_online.log, trace_repo_online/
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parity  # noqa: E402
import probe  # noqa: E402

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
CONTROL = Path(__file__).resolve().parent
REPO = parity.REPO


def serve_online(tag="repo_online", download_dir=None):
    """With download_dir, the default Hub cache is an empty directory and the weights come from download_dir."""
    deploy = probe.site_pkg() / "deploy" / f"moshi_{tag}.yaml"
    deploy.write_text("base_config: personaplex.yaml\nduplex_session:\n  max_sessions: 2\n")
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
    env |= {"PYTHONPATH": str(CONTROL / "inject"), "MOSHI_TRACE_DIR": str(OUT / f"trace_{tag}")}
    extra = []
    if download_dir:
        empty = Path("/tmp/empty-hub-cache")
        empty.mkdir(exist_ok=True)
        env["HF_HUB_CACHE"] = str(empty)
        extra = ["--download-dir", download_dir]
    log = open(OUT / f"server_{tag}.log", "w")
    cmd = ["vllm", "serve", REPO, "--omni", "--port", str(parity.PORT), "--deploy-config", str(deploy), *extra]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 1200:
        if proc.poll() is not None:
            raise RuntimeError(f"{tag}: server exited with {proc.returncode}")
        try:
            if httpx.get(f"http://127.0.0.1:{parity.PORT}/v1/models", timeout=5).status_code == 200:
                print(f"{tag}: ready after {time.time() - t0:.0f}s", flush=True)
                return proc, log
        except Exception:
            pass
        time.sleep(5)
    os.killpg(proc.pid, signal.SIGKILL)
    raise RuntimeError(f"{tag}: server not ready")


async def session_with_errors(pcm, tag):
    """parity.one_session plus the error events, so a failed open is recorded."""
    import base64

    import numpy as np
    import soundfile as sf

    from vllm_omni.clients.duplex import AudioFormat, DuplexClient, SessionConfig

    config = SessionConfig(input_audio=AudioFormat("pcm_f32le", 24_000), output_audio=AudioFormat("pcm16", 24_000))
    url = f"ws://127.0.0.1:{parity.PORT}/v1/realtime?duplex=1"
    audio, types, errors = bytearray(), {}, []
    async with DuplexClient(url, model=REPO, config=config) as client:
        async def reader():
            async for event in client.events():
                raw = event.raw if hasattr(event, "raw") else {}
                etype = raw.get("type")
                types[etype] = types.get(etype, 0) + 1
                if etype in ("response.output_audio.delta", "response.audio.delta"):
                    delta = raw.get("delta") or raw.get("audio")
                    if isinstance(delta, str):
                        audio.extend(base64.b64decode(delta))
                elif etype == "error":
                    errors.append(raw)

        task = asyncio.create_task(reader())
        try:
            await client.stream_pcm(pcm.astype("<f4").tobytes(), chunk_ms=80, realtime=True)
        except Exception as e:  # the server may close the socket on a failed open
            errors.append({"client_exception": repr(e)})
        await asyncio.sleep(3.0)
        task.cancel()
    pcm16 = np.frombuffer(bytes(audio), dtype="<i2")
    sf.write(OUT / f"vllm_{tag}.wav", pcm16.astype(np.float32) / 32768.0, 24000)
    return {"audio_s": len(pcm16) / 24000, "events": types, "errors": errors}


def main():
    import argparse

    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", default=None, help="serve with --download-dir and an empty default cache")
    args = ap.parse_args()
    tag = "repo_download_dir" if args.download_dir else "repo_online"

    probe.sh("nvidia-smi --query-gpu=name,driver_version --format=csv", log="gpu.txt")
    probe.step_download()  # fills the Hub cache the online server reads
    inp = probe.step_input()
    pcm, _ = sf.read(inp, dtype="float32")
    result = {"hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"), "download_dir": args.download_dir}
    proc, log = serve_online(tag, args.download_dir)
    try:
        result["client"] = asyncio.run(session_with_errors(pcm[: 24000 * 10], tag))
        time.sleep(2)
    finally:
        parity.stop(proc, log)
    server_log = (OUT / f"server_{tag}.log").read_text(errors="replace")
    result["model_id_replaced_by_path"] = "replace model_id" in server_log
    result["prefill_unavailable_in_log"] = "prefill_unavailable" in server_log
    result["gated_repo_requests_in_log"] = server_log.count("personaplex-7b-v1")
    runtimes = []
    for f in sorted((OUT / f"trace_{tag}").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("ev") == "runtime":
                runtimes.append({"moshi": r.get("moshi"), "model_path": r.get("model_path")})
    result["stage0_runtimes"] = runtimes
    (OUT / f"{tag}.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()

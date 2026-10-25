"""F0 probe: can upstream vLLM-Omni serve kyutai/moshiko-pytorch-bf16 on the PersonaPlex pipeline?

Steps (outputs in $OUT_DIR):
  1. download the ungated moshiko repo into the HF cache and list its weight keys
  2. build a 24 kHz input from LibriSpeech dummy utterances plus trailing silence
  3. official moshi reference, greedy and default sampling (ref_moshi.py)
  4. start `vllm serve <moshiko> --omni` on an overlay of personaplex.yaml and,
     if it comes up, run tests/e2e/online_serving/personaplex_realtime_duplex.py;
     record where it fails
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
CONTROL = Path(__file__).resolve().parent
SRC = Path("/opt/vllm-omni")
REPO = "kyutai/moshiko-pytorch-bf16"
PORT = 8099


def sh(cmd, timeout=None, log=None):
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=timeout)
    if log:
        (OUT / log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    return r


def site_pkg() -> Path:
    out = subprocess.run([sys.executable, "-c", "import vllm_omni, os; print(os.path.dirname(vllm_omni.__file__))"],
                         text=True, capture_output=True, cwd="/", check=True)
    return Path(out.stdout.strip())


def step_download():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download(REPO)
    info = {"path": path, "files": sorted(os.listdir(path))}
    with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
        keys = list(f.keys())
        info["n_keys"] = len(keys)
        info["shapes"] = {k: list(f.get_slice(k).get_shape()) for k in keys if not k.startswith("transformer.layers.")
                          or k.startswith("transformer.layers.0.")}
    (OUT / "moshiko_keys.json").write_text(json.dumps(info, indent=2))
    print(f"moshiko at {path}: {info['n_keys']} keys", flush=True)
    return path


def step_input():
    import io

    import numpy as np
    import soundfile as sf
    import soxr
    from datasets import Audio, load_dataset

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    pieces = []
    for row in list(ds)[:3]:
        a, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        pieces += [soxr.resample(a, sr, 24000).astype(np.float32), np.zeros(12000, np.float32)]
    pieces.append(np.zeros(24000 * 6, np.float32))
    pcm = np.concatenate(pieces)
    sf.write(OUT / "input_24k.wav", pcm, 24000)
    print(f"input {len(pcm) / 24000:.1f}s", flush=True)
    return OUT / "input_24k.wav"


def step_ref(inp):
    for flag, tag in (("--semantics server", "server_greedy"), ("--semantics inference", "inference_greedy"),
                      ("--semantics server --sampling", "server_sampled")):
        t0 = time.time()
        r = sh(f"{sys.executable} {CONTROL / 'ref_moshi.py'} --repo {REPO} --input {inp} --out-dir {OUT} {flag}",
               timeout=1800, log=f"ref_{tag}.log")
        print(f"ref {tag}: rc={r.returncode} {time.time() - t0:.0f}s", flush=True)
        print(r.stdout[-1500:], flush=True)


def step_serve(model_path, inp):
    deploy = site_pkg() / "deploy" / "moshi_probe.yaml"
    deploy.write_text("base_config: personaplex.yaml\nduplex_session:\n  max_sessions: 2\n")
    log = open(OUT / "server.log", "w")
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    cmd = ["vllm", "serve", model_path, "--omni", "--port", str(PORT), "--deploy-config", str(deploy)]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    status = {"ready": False}
    t0 = time.time()
    try:
        while time.time() - t0 < 1200:
            if proc.poll() is not None:
                status["exit"] = proc.returncode
                break
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/v1/models", timeout=5).status_code == 200:
                    status["ready"] = True
                    break
            except Exception:
                pass
            time.sleep(5)
        status["boot_s"] = time.time() - t0
        print(f"serve status {status}", flush=True)
        if status["ready"]:
            r = sh(f"{sys.executable} {SRC}/tests/e2e/online_serving/personaplex_realtime_duplex.py --model {model_path} "
                   f"--input-wav {inp} --output-dir {OUT}/realtime --url ws://127.0.0.1:{PORT}/v1/realtime?duplex=1",
                   timeout=900, log="realtime.log")
            status["realtime_rc"] = r.returncode
            print(r.stdout[-2000:], r.stderr[-3000:], flush=True)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        sh("pkill -9 -f 'vllm serve' || true")
        log.close()
    (OUT / "serve_status.json").write_text(json.dumps(status, indent=2))
    tail = (OUT / "server.log").read_text().splitlines()
    errs = [ln for ln in tail if any(s in ln for s in ("Error", "error", "Traceback", "raise", "FileNotFound"))]
    print("\n".join(errs[-40:]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="download,input,ref,serve")
    args = ap.parse_args()
    sh("nvidia-smi --query-gpu=name,driver_version --format=csv", log="gpu.txt")
    sh(f"{sys.executable} -c 'import vllm, vllm_omni, torch, moshi; print(vllm.__version__, torch.__version__, moshi.__version__)'",
       log="versions.txt")
    steps = args.steps.split(",")
    model_path = step_download() if "download" in steps else None
    inp = step_input() if "input" in steps else OUT / "input_24k.wav"
    if "ref" in steps:
        step_ref(inp)
    if "serve" in steps:
        step_serve(model_path, inp)


if __name__ == "__main__":
    main()

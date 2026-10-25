"""Moshi on the other deploy and checkpoint variants, one paced session each.

  mrv2     kyutai/moshiko-pytorch-bf16 on personaplex_duplex_mrv2.yaml
  moshika  kyutai/moshika-pytorch-bf16 on personaplex.yaml, plus its own greedy reference

Each records the Stage 0 steps with the probe and compares them with the reference
of the same checkpoint (agreement as in parity2). Outputs in $OUT_DIR/variants.json.
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
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parity  # noqa: E402
import parity2  # noqa: E402
import probe  # noqa: E402

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
CONTROL = Path(__file__).resolve().parent


def serve(model_path, base, tag):
    deploy = probe.site_pkg() / "deploy" / f"moshi_{tag}.yaml"
    deploy.write_text(f"base_config: {base}\n")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(CONTROL / "inject"),
           "MOSHI_TRACE_DIR": str(OUT / f"trace_{tag}")}
    log = open(OUT / f"server_{tag}.log", "w")
    cmd = ["vllm", "serve", model_path, "--omni", "--port", str(parity.PORT), "--deploy-config", str(deploy)]
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


def main():
    from huggingface_hub import snapshot_download
    import soundfile as sf

    probe.sh("nvidia-smi --query-gpu=name,driver_version --format=csv", log="gpu.txt")
    inp = probe.step_input()
    pcm, _ = sf.read(inp, dtype="float32")
    results = {}
    for tag, repo, base in (("mrv2", "kyutai/moshiko-pytorch-bf16", "personaplex_duplex_mrv2.yaml"),
                            ("moshika", "kyutai/moshika-pytorch-bf16", "personaplex.yaml")):
        model_path = snapshot_download(repo)
        d = OUT / f"ref_{tag}"
        d.mkdir(exist_ok=True)
        r = probe.sh(f"{sys.executable} {CONTROL / 'ref_moshi.py'} --repo {repo} --input {inp} --out-dir {d} "
                     "--semantics server", timeout=1800, log=f"ref_{tag}.log")
        ref = dict(np.load(d / "ref_server_greedy.npz"))
        res = {"ref_rc": r.returncode, "ref_text": json.loads((d / "ref_server_greedy.json").read_text())["text"]}
        try:
            proc, log = serve(model_path, base, tag)
            try:
                res["client"] = asyncio.run(parity.one_session(model_path, pcm, tag))
                time.sleep(2)
            finally:
                parity.stop(proc, log)
            text, codes = parity.load_steps(OUT / f"trace_{tag}")
            np.savez(OUT / f"vllm_{tag}.npz", text=text, codes=codes)
            vt, vc = parity2.vllm_records(text, codes)
            res["vllm_vs_ref"] = parity2.agreement(ref["text"], ref["codes"], vt, vc)
        except Exception as e:
            res["error"] = repr(e)
        results[tag] = res
        print(f"{tag}: {json.dumps(res, default=str)[:1500]}", flush=True)
    results["asr"] = parity2.asr({tag: OUT / f"vllm_{tag}.wav" for tag in ("mrv2", "moshika")}
                                 | {f"ref_{tag}": OUT / f"ref_{tag}" / "ref_server_greedy.wav" for tag in ("mrv2", "moshika")})
    (OUT / "variants.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()

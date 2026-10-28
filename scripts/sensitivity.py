"""How sensitive is Moshi's greedy second-reply timing to numerics alone? (#9319 open difference)

Every arm runs in one container on the same input (probe.step_input, as x2-evidence):
  ref_bf16_a/b      kyutai moshi 0.2.13, bf16, batch 1 (the reference as before)
  ref_noreduced     bf16 with cuBLAS reduced-precision bf16 reduction off
  ref_b2 / ref_b4   bf16, the same input on every row of a batch of 2 / 4 (row 0 kept)
  ref_fp32          fp32 weights and activations
  vllm_graphs       this branch, deploy default (personaplex.yaml), one session

Only the numerics change between the reference arms. For each arm: the step of the first
text token of the second reply (first non-pad text step after step 40), the first-reply
steps, and agreement with ref_bf16_a. Output: $OUT_DIR/sensitivity.json.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parity  # noqa: E402
import parity2  # noqa: E402
import probe  # noqa: E402

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
CONTROL = Path(__file__).resolve().parent
MOSHIKO = "kyutai/moshiko-pytorch-bf16"
VARIANTS = {
    "ref_bf16_a": "",
    "ref_bf16_b": "",
    "ref_noreduced": "--no-reduced-bf16",
    "ref_b2": "--batch 2",
    "ref_b4": "--batch 4",
    "ref_fp32": "--dtype fp32",
}


def run_variant(inp, tag, flags):
    d = OUT / tag
    d.mkdir(exist_ok=True)
    r = probe.sh(f"{sys.executable} {CONTROL / 'ref_variant.py'} --repo {MOSHIKO} --input {inp} --out-dir {d} "
                 f"--semantics server {flags}", timeout=1800, log=f"{tag}.log")
    print(f"{tag}: rc={r.returncode}", flush=True)
    return dict(np.load(d / "ref_server_greedy.npz"))


def reply_steps(text):
    nonpad = [int(i) for i in np.nonzero((text != 0) & (text != 3))[0]]
    first = [i for i in nonpad if i <= 40]
    later = [i for i in nonpad if i > 40]
    return {"first_reply_steps": first, "second_reply_start": later[0] if later else None,
            "second_reply_tokens": len(later)}


def main():
    from huggingface_hub import snapshot_download
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--src-sha", required=True)
    args = ap.parse_args()
    probe.sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv", log="gpu.txt")
    inp = probe.step_input()
    pcm, _ = sf.read(inp, dtype="float32")
    path = snapshot_download(MOSHIKO)
    arms = {}
    for tag, flags in VARIANTS.items():
        try:
            arms[tag] = run_variant(inp, tag, flags)
        except Exception as e:
            print(f"{tag}: {e!r}", flush=True)

    try:
        proc, log = parity.serve(path, "graphs", 2)
        try:
            asyncio.run(parity.one_session(path, pcm, "graphs"))
            time.sleep(2)
        finally:
            parity.stop(proc, log)
        text, codes = parity.load_steps(OUT / "trace_graphs")
        vt, vc = parity2.vllm_records(text, codes)
        arms["vllm_graphs"] = {"text": vt, "codes": vc}
        np.savez(OUT / "vllm_graphs.npz", text=vt, codes=vc)
    except Exception as e:
        print(f"vllm_graphs: {e!r}", flush=True)

    base = arms.get("ref_bf16_a")
    summary = {"vllm_omni_revision": args.src_sha, "checkpoint": Path(path).name, "arms": {}}
    for tag, rec in arms.items():
        row = reply_steps(rec["text"])
        if base is not None and tag != "ref_bf16_a":
            agree = parity2.agreement(base["text"], base["codes"], rec["text"], rec["codes"])
            row |= {k: agree[k] for k in ("steps", "text_match", "first_text_diff", "frame_match", "first_frame_diff")}
        summary["arms"][tag] = row
        print(f"{tag}: {json.dumps(row)}", flush=True)
    (OUT / "sensitivity.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

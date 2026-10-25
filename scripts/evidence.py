"""Moshi parity evidence on one revision, every arm in one container (for the #9319 review).

  references   two official greedy references per checkpoint (moshi.server framing), same input
  eager        moshiko, local snapshot, depformer CUDA graphs off
  graphs       moshiko, local snapshot, deploy default (personaplex.yaml)
  mrv2         moshiko, local snapshot, personaplex_duplex_mrv2.yaml
  repo_online  moshiko served by repo id with the Hub online (HF_HUB_OFFLINE unset)
  repo_download_dir  the same with --download-dir and an empty default Hub cache
  moshika      moshika, local snapshot, deploy default

Every arm is compared with both references of its checkpoint, so the reference's own
spread in this container is the floor for every arm. Outputs in $OUT_DIR:
summary.json, ref_*/, vllm_<arm>.{npz,wav}, server_<arm>.log, trace_<arm>/, asr.
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
import repo_online  # noqa: E402
import variants  # noqa: E402

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
CONTROL = Path(__file__).resolve().parent
MOSHIKO = "kyutai/moshiko-pytorch-bf16"
MOSHIKA = "kyutai/moshika-pytorch-bf16"


def run_ref(repo, inp, tag):
    d = OUT / tag
    d.mkdir(exist_ok=True)
    r = probe.sh(f"{sys.executable} {CONTROL / 'ref_moshi.py'} --repo {repo} --input {inp} --out-dir {d} "
                 "--semantics server", timeout=1800, log=f"{tag}.log")
    print(f"{tag}: rc={r.returncode}", flush=True)
    return dict(np.load(d / "ref_server_greedy.npz"))


def compare_arm(tag, refs):
    text, codes = parity.load_steps(OUT / f"trace_{tag}")
    np.savez(OUT / f"vllm_{tag}.npz", text=text, codes=codes)
    if len(text) < 2:
        return {"steps": int(len(text))}
    vt, vc = parity2.vllm_records(text, codes)
    return {name: parity2.agreement(ref["text"], ref["codes"], vt, vc) for name, ref in refs.items()}


def runtimes(tag):
    out = []
    for f in sorted((OUT / f"trace_{tag}").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("ev") == "runtime":
                out.append({"moshi": r.get("moshi"), "model_path": r.get("model_path")})
    return out


def main():
    from huggingface_hub import snapshot_download
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--src-sha", required=True)
    args = ap.parse_args()
    probe.sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv", log="gpu.txt")
    inp = probe.step_input()
    pcm, _ = sf.read(inp, dtype="float32")
    paths = {repo: snapshot_download(repo) for repo in (MOSHIKO, MOSHIKA)}
    summary = {
        "vllm_omni_revision": args.src_sha,
        "checkpoints": {repo: Path(p).name for repo, p in paths.items()},
        "input": "librispeech_asr_dummy clean/validation 1272-128104-0000..0002 at 24 kHz, 0.5 s gaps, 6 s tail",
    }

    refs = {"moshiko": {}, "moshika": {}}
    for ckpt, repo in (("moshiko", MOSHIKO), ("moshika", MOSHIKA)):
        for tag in ("a", "b"):
            refs[ckpt][f"ref_{tag}"] = run_ref(repo, inp, f"ref_{ckpt}_{tag}")
        a, b = refs[ckpt]["ref_a"], refs[ckpt]["ref_b"]
        summary[f"{ckpt}_ref_a_vs_ref_b"] = parity2.agreement(a["text"], a["codes"], b["text"], b["codes"])
        print(f"{ckpt} ref vs ref: {json.dumps(summary[f'{ckpt}_ref_a_vs_ref_b'])}", flush=True)

    arms = {}
    for tag, ckpt in (("eager", "moshiko"), ("graphs", "moshiko"), ("mrv2", "moshiko"),
                      ("repo_online", "moshiko"), ("repo_download_dir", "moshiko"), ("moshika", "moshika")):
        res = {}
        try:
            if tag in ("eager", "graphs"):
                proc, log = parity.serve(paths[MOSHIKO], tag, 2)
                client = lambda: parity.one_session(paths[MOSHIKO], pcm, tag)  # noqa: E731
            elif tag == "mrv2":
                proc, log = variants.serve(paths[MOSHIKO], "personaplex_duplex_mrv2.yaml", tag)
                client = lambda: parity.one_session(paths[MOSHIKO], pcm, tag)  # noqa: E731
            elif tag == "moshika":
                proc, log = variants.serve(paths[MOSHIKA], "personaplex.yaml", tag)
                client = lambda: parity.one_session(paths[MOSHIKA], pcm, tag)  # noqa: E731
            else:
                # The Volume's hub cache already holds the weights, so --download-dir finds them there.
                download_dir = str(Path(paths[MOSHIKO]).parents[2]) if tag == "repo_download_dir" else None
                proc, log = repo_online.serve_online(tag, download_dir)
                res["download_dir"] = download_dir
                client = lambda: repo_online.session_with_errors(pcm, tag)  # noqa: E731
            try:
                res["client"] = asyncio.run(client())
                time.sleep(2)
            finally:
                parity.stop(proc, log)
            res["vs"] = compare_arm(tag, refs[ckpt])
            res["stage0_runtimes"] = runtimes(tag)
            if tag.startswith("repo_"):
                server_log = (OUT / f"server_{tag}.log").read_text(errors="replace")
                res["model_id_replaced_by_path"] = "replace model_id" in server_log
                res["gated_repo_requests_in_log"] = server_log.count("personaplex-7b-v1")
        except Exception as e:
            res["error"] = repr(e)
        arms[tag] = res
        brief = {k: (v.get("text_match"), v.get("frame_match")) for k, v in res.get("vs", {}).items() if isinstance(v, dict)}
        print(f"{tag}: text/frame vs refs {brief} runtimes {res.get('stage0_runtimes')} error {res.get('error')}",
              flush=True)
    summary["arms"] = arms

    wavs = {f"ref_{c}_{t}": OUT / f"ref_{c}_{t}" / "ref_server_greedy.wav" for c in refs for t in ("a", "b")}
    wavs |= {f"vllm_{tag}": OUT / f"vllm_{tag}.wav" for tag in arms}
    try:
        summary["asr"] = parity2.asr(wavs)
    except Exception as e:
        summary["asr_error"] = repr(e)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

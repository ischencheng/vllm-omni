"""Reference run of the official kyutai `moshi` package with a numeric variant (measurement only).

Same loop as ref_moshi.py; --dtype, --batch (row 0 is recorded, every row gets the
same input) and --no-reduced-bf16 change only the numerics of the reference.

python ref_moshi.py --repo kyutai/moshiko-pytorch-bf16 --input in24k.wav --out-dir DIR [--sampling]

Two framings of the same loop:
  server     moshi/server.py: one LMGen step per 1920-sample frame; the first
             frame lands on the initial-token step and Mimi's encoder is reset
             after it (default; this is what a live duplex session does)
  inference  moshi/run_inference.py: the first frame is stepped twice
Records every emitted step's text token and the 8 agent audio codes.
Writes ref_<semantics>_<greedy|sampled>.npz/.json/.wav.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="kyutai/moshiko-pytorch-bf16")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sampling", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--semantics", choices=("server", "inference"), default="server")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--no-reduced-bf16", action="store_true")
    args = ap.parse_args()
    if args.no_reduced_bf16:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    from moshi.models import LMGen, loaders

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    info = loaders.CheckpointInfo.from_hf_repo(args.repo)
    mimi = info.get_mimi(device="cuda")
    lm = info.get_moshi(device="cuda", dtype=torch.float32 if args.dtype == "fp32" else torch.bfloat16)
    tok = info.get_text_tokenizer()
    meta = {"lm_config": {k: v for k, v in (info.lm_config or {}).items() if isinstance(v, (int, float, str, list, bool))}}
    lm_gen = LMGen(lm, use_sampling=args.sampling)
    meta.update(temp=lm_gen.temp, temp_text=lm_gen.temp_text, top_k=lm_gen.top_k, top_k_text=lm_gen.top_k_text,
                use_sampling=lm_gen.use_sampling, delays=list(lm.delays), dep_q=lm.dep_q, n_q=lm.n_q)
    frame = int(mimi.sample_rate / mimi.frame_rate)
    pcm, sr = sf.read(args.input, dtype="float32")
    assert sr == 24000, sr
    x = torch.from_numpy(pcm).to("cuda")[None, None].repeat(args.batch, 1, 1)
    chunks = [c for c in x.split(frame, dim=2) if c.shape[-1] == frame]
    mimi.streaming_forever(args.batch)
    lm_gen.streaming_forever(args.batch)
    texts, codes, user_codes, pcms = [], [], [], []
    t0 = time.time()
    first = True
    with torch.no_grad():
        for chunk in chunks:
            c = mimi.encode(chunk)
            user_codes.append(c[0, :, 0].cpu().numpy())
            if first:
                first = False
                if args.semantics == "inference":
                    assert lm_gen.step(c) is None
                else:
                    mimi.reset_streaming()
            tokens = lm_gen.step(c)
            if tokens is None:
                continue
            texts.append(int(tokens[0, 0, 0]))
            codes.append(tokens[0, 1:, 0].cpu().numpy())
            pcms.append(mimi.decode(tokens[:, 1:]).cpu().numpy()[0, 0])
    dt = time.time() - t0
    tag = f"{args.semantics}_{'sampled' if args.sampling else 'greedy'}"
    text = "".join(tok.id_to_piece(t).replace("▁", " ") for t in texts if t not in (0, 3))
    np.savez(out / f"ref_{tag}.npz", text=np.array(texts), codes=np.stack(codes), user_codes=np.stack(user_codes))
    sf.write(out / f"ref_{tag}.wav", np.concatenate(pcms), 24000)
    meta.update(dtype=args.dtype, batch=args.batch, no_reduced_bf16=args.no_reduced_bf16, semantics=args.semantics, frames_in=len(chunks), steps_out=len(texts), seconds=dt, ms_per_step=1e3 * dt / max(1, len(texts)),
                text=text)
    (out / f"ref_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("frames_in", "steps_out", "ms_per_step", "text")}, indent=2))


if __name__ == "__main__":
    main()

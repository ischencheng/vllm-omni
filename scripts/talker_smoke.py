"""MiniCPM-o 4.5 turn MRv2 Talker: no draft vs DSpark draft on stage 1, one container.

python talker_smoke.py [--arms off,dspark] [--rounds 2] [--conc 1,16] [--greedy] [--k 7]
-> $OUT_DIR/<arm>-r<round>/server.log, results.json, c<conc>_<i>.wav; $OUT_DIR/summary.json

Each arm starts `vllm serve openbmb/MiniCPM-o-4_5 --omni` with an overlay of
minicpmo_4_5_turn_mrv2.yaml. The dspark arm only adds a stage-1
speculative_config; the Thinker has no draft in either arm. Arms alternate per
round. Requests stream text + audio; per request: first audio chunk (TTFA),
end-to-end latency, audio seconds. Whisper (openai/whisper-small.en) scores the
audio against the reply text. --greedy sets the Talker temperature to 0 so the
codes of the two arms can be compared (the debug tree dumps them to
$SPEC_DUMP_CODES).
"""

import argparse
import base64
import io
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
MODEL = "openbmb/MiniCPM-o-4_5"
DRAFT = "Qun1023/MiniCPM-o-4_5-talker-DSpark"
DRAFT_REV = "0457ffd6980c2d1f19ebd3f1775f49764fa821af"
PORT = 8091

PROMPTS = [
    "What is the capital of China? Answer in one sentence and then name three famous places there.",
    "Explain how a hash map handles collisions. Keep it under 60 words.",
    "List the planets of the solar system in order from the sun.",
    "A train travels 180 km in 2 hours. What is its average speed?",
    "Write a short story of about 50 words about a robot that learns to paint.",
    "What are the main differences between TCP and UDP? Answer in two sentences.",
    "Give me a recipe for scrambled eggs in three short steps.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Explain the difference between supervised and unsupervised learning in two sentences.",
    "What is 17 multiplied by 23?",
    "Describe the water cycle in simple terms, in two sentences.",
    "Why is the sky blue? Answer in two sentences.",
    "Tell me a fun fact about octopuses.",
    "How do vaccines work? Keep it short.",
    "What should I pack for a weekend hiking trip?",
    "Recommend a good habit for better sleep and explain why in one sentence.",
]

BASE = "minicpmo_4_5_turn_mrv2.yaml"


def words(text):
    return re.sub(r"[^a-z0-9' ]", " ", (text or "").lower().replace("-", " ")).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def deploy_dir():
    import vllm_omni

    return Path(vllm_omni.__file__).resolve().parent / "deploy"


LIFTED = ("markov_rank", "markov_head_type", "enable_confidence_head", "confidence_head_with_markov",
          "confidence_head_alpha", "mask_token_id", "target_layer_ids")


def stage_draft():
    """Local copy of the draft that vLLM 0.32 can load as a Talker drafter.

    * config.json: the DSpark fields also at the top level. Qwen3DSparkModel
      reads markov_rank (and the confidence-head switches) there; this
      checkpoint keeps them only under dflash_config (the Thinker draft has both).
    * weights: the Talker's codec table (embed_tokens) and head (lm_head) added.
      The checkpoint ships neither, and vLLM shares a target's only when the
      draft vocabulary is not larger than the stage's configured vocabulary.
    """
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    dst = Path("/work/talker-draft")
    if (dst / "model.safetensors").exists():
        return dst
    src = Path(snapshot_download(DRAFT, revision=DRAFT_REV))
    dst.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    for key in LIFTED:
        config.setdefault(key, config["dflash_config"][key])
    (dst / "config.json").write_text(json.dumps(config, indent=2))

    index = json.loads(Path(hf_hub_download(MODEL, "model.safetensors.index.json")).read_text())["weight_map"]
    names = ("tts.emb_code.0.weight", "tts.head_code.0.parametrizations.weight.original0",
             "tts.head_code.0.parametrizations.weight.original1")
    talker = {}
    for name in names:
        with safe_open(hf_hub_download(MODEL, index[name]), framework="pt") as f:
            talker[name] = f.get_tensor(name)
    weights = load_file(src / "model.safetensors")
    weights["embed_tokens.weight"] = talker[names[0]].to(torch.bfloat16).contiguous()
    # As the Talker restores its weight-normed head (dim 0), then casts to bf16.
    head = torch._weight_norm(talker[names[2]], talker[names[1]], dim=0)
    weights["lm_head.weight"] = head.to(torch.bfloat16).contiguous()
    save_file(weights, dst / "model.safetensors", metadata={"format": "pt"})
    print(f"staged draft: {len(weights)} tensors, embed {tuple(weights['embed_tokens.weight'].shape)}, "
          f"head {tuple(weights['lm_head.weight'].shape)}", flush=True)
    return dst


def write_overlay(arm, args):
    lines = [f"base_config: {BASE}", "stages:", "  - stage_id: 1"]
    if arm == "dspark":
        lines += [
            "    speculative_config:",
            "      method: dspark",
            f"      model: {stage_draft()}",
            f"      num_speculative_tokens: {args.k}",
        ]
        if args.draft_sample:
            lines += [f"      draft_sample_method: {args.draft_sample}"]
    if args.greedy:
        lines += [
            "    default_sampling_params:",
            "      temperature: 0.0",
            "      top_p: 1.0",
            "      top_k: -1",
            "      repetition_penalty: 1.05",
            "      min_tokens: 50",
            "      max_tokens: 4096",
            "      seed: 42",
            "      detokenize: false",
        ]
    if args.unseeded:
        # The stage default (seed 42 for every request) without the seed: each
        # request draws its own codes, so arms can be compared by distribution.
        lines += [
            "    default_sampling_params:",
            "      temperature: 0.8",
            "      top_p: 0.85",
            "      top_k: 25",
            "      repetition_penalty: 1.05",
            "      min_tokens: 50",
            "      max_tokens: 4096",
            "      seed: null",
            "      detokenize: false",
        ]
    if len(lines) == 3:
        lines += ["    max_num_seqs: 8"]
    path = deploy_dir() / f"_talker_{arm}.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def wait_ready(proc, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False, time.time() - t0
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                return True, time.time() - t0
        except requests.RequestException:
            pass
        time.sleep(3)
    return False, time.time() - t0


def one_request(prompt, max_tokens):
    import soundfile as sf

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "modalities": ["text", "audio"],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 42,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False, "use_tts_template": True},
    }
    t0 = time.perf_counter()
    row = {"prompt": prompt}
    text, pcm, rate, chunk_ids = [], [], None, set()
    try:
        with requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body, stream=True, timeout=900) as r:
            row["status"] = r.status_code
            if r.status_code != 200:
                row["error"] = r.text[:2000]
                return row
            for raw in r.iter_lines():
                if not raw or not raw.startswith(b"data: "):
                    continue
                data = raw[6:]
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                chunk_ids.add(chunk.get("id"))
                for choice in chunk.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if not content:
                        continue
                    if chunk.get("modality") == "audio":
                        if "ttfa_s" not in row:
                            row["ttfa_s"] = time.perf_counter() - t0
                        wav, rate = sf.read(io.BytesIO(base64.b64decode(content)), dtype="float32")
                        pcm.append(wav if wav.ndim == 1 else wav.mean(axis=1))
                        row["audio_chunks"] = row.get("audio_chunks", 0) + 1
                    else:
                        if "ttft_s" not in row:
                            row["ttft_s"] = time.perf_counter() - t0
                        text.append(content)
    except Exception as e:  # noqa: BLE001
        row["error"] = repr(e)[:2000]
    row["latency_s"] = time.perf_counter() - t0
    row["text"] = "".join(text)
    row["ids"] = sorted(i for i in chunk_ids if i)
    if pcm:
        import numpy as np

        row["pcm"] = np.concatenate(pcm)
        row["rate"] = rate
        row["audio_s"] = len(row["pcm"]) / rate
    return row


def run_batch(prompts, max_tokens, conc):
    rows = [None] * len(prompts)
    t0 = time.perf_counter()
    if conc <= 1:
        for i, p in enumerate(prompts):
            rows[i] = one_request(p, max_tokens)
    else:
        sem = threading.Semaphore(conc)

        def work(i, p):
            with sem:
                rows[i] = one_request(p, max_tokens)

        threads = [threading.Thread(target=work, args=(i, p)) for i, p in enumerate(prompts)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    return rows, time.perf_counter() - t0


def run_arm(arm, rnd, args):
    import soundfile as sf

    out = OUT / f"{arm}-r{rnd}"
    out.mkdir(parents=True, exist_ok=True)
    overlay = write_overlay(arm, args)
    (out / "overlay.yaml").write_text(overlay.read_text())
    log = open(out / "server.log", "w")
    template = deploy_dir().parent / "transformers_utils" / "chat_templates" / "minicpmo45_native.jinja"
    cmd = [sys.executable, "-m", "vllm_omni.entrypoints.cli.main", "serve", MODEL, "--omni",
           "--host", "127.0.0.1", "--port", str(PORT), "--deploy-config", str(overlay),
           "--trust-remote-code", "--chat-template", str(template), "--chat-template-content-format", "openai"]
    env = dict(os.environ, SPEC_DUMP_CODES=str(out / "codes.jsonl"))
    print(f"[{arm}-r{rnd}] {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    result = {"arm": arm, "round": rnd, "cmd": cmd}
    try:
        ok, startup = wait_ready(proc, args.startup_timeout)
        result["startup_s"] = startup
        result["ready"] = ok
        print(f"[{arm}-r{rnd}] ready={ok} after {startup:.0f}s", flush=True)
        if not ok:
            log.flush()
            lines = (out / "server.log").read_text(errors="replace").splitlines()
            first = next((i for i, ln in enumerate(lines) if "Traceback" in ln), max(0, len(lines) - 40))
            for ln in lines[first : first + 70]:
                print("  | " + re.sub(r"^.*?\[stage_engine_core_proc\.py:\d+\]\S* ", "", ln)[:300], flush=True)
            return result
        # Warm-up (not timed).
        run_batch(PROMPTS[:4], 48, 4)
        for conc in args.conc:
            prompts = PROMPTS[: args.num_prompts] if conc == 1 else (PROMPTS * 4)[: conc * args.c16_waves]
            # SPECACC lines (debug tree) written before this phase starts.
            log.flush()
            result.setdefault("specacc_marks", {})[f"c{conc}"] = (out / "server.log").read_text(
                errors="replace").count("SPECACC")
            rows, wall = run_batch(prompts, args.max_tokens, conc)
            for i, r in enumerate(rows):
                pcm = r.pop("pcm", None)
                if pcm is not None:
                    path = out / f"c{conc}_{i:02d}.wav"
                    sf.write(path, pcm, r["rate"])
                    r["audio_file"] = path.name
            ok_rows = [r for r in rows if not r.get("error") and r.get("audio_s")]
            audio_s = sum(r["audio_s"] for r in ok_rows)
            result[f"c{conc}"] = {
                "wall_s": wall,
                "requests": len(rows),
                "ok": len(ok_rows),
                "req_per_s": len(ok_rows) / wall if wall else None,
                "audio_s_per_s": audio_s / wall if wall else None,
                "rows": rows,
            }
            print(f"[{arm}-r{rnd}] c{conc}: {len(ok_rows)}/{len(rows)} ok in {wall:.1f}s, "
                  f"{audio_s:.1f} s audio", flush=True)
            for r in [r for r in rows if r.get("error") or not r.get("audio_s")][:2]:
                print(f"  error: {r.get('status')} {str(r.get('error', ''))[:500]}", flush=True)
        try:
            (out / "metrics.txt").write_text(requests.get(f"http://127.0.0.1:{PORT}/metrics", timeout=10).text)
        except requests.RequestException as e:
            (out / "metrics.txt").write_text(f"metrics unavailable: {e}\n")
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        log.close()
        time.sleep(5)
        text = (out / "server.log").read_text(errors="replace")
        result["error_tail"] = [ln for ln in text.splitlines() if re.search(r"(?i)error|exception|traceback", ln)][-40:]
        (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def score_audio():
    import soundfile as sf
    from transformers import pipeline

    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device=0)
    scores = []
    for res_file in sorted(OUT.glob("*-r*/results.json")):
        res = json.loads(res_file.read_text())
        for key, val in res.items():
            if not re.fullmatch(r"c\d+", key):
                continue
            for i, r in enumerate(val["rows"]):
                if not r.get("audio_file"):
                    continue
                pcm, rate = sf.read(res_file.parent / r["audio_file"], dtype="float32")
                hyp = pipe({"raw": pcm, "sampling_rate": rate}, chunk_length_s=30, batch_size=4)["text"].strip()
                r["asr"] = hyp
                r["wer"] = wer(r.get("text"), hyp)
                scores.append({"arm": res["arm"], "round": res["round"], "conc": key, "idx": i, "wer": r["wer"],
                               "audio_s": r["audio_s"]})
        res_file.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="off,dspark")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--conc", default="1,16")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--c16-waves", type=int, default=2, help="requests per concurrency level = conc x waves")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--unseeded", action="store_true", help="Talker defaults without the per-request seed")
    ap.add_argument("--draft-sample", default="", help="speculative_config.draft_sample_method (vLLM default: greedy)")
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--startup-timeout", type=int, default=2400)
    ap.add_argument("--env", default="", help="comma-separated KEY=VALUE for the server")
    args = ap.parse_args()
    for kv in filter(None, args.env.split(",")):
        k, v = kv.split("=", 1)
        os.environ[k] = v
    args.conc = [int(c) for c in args.conc.split(",")]
    arms = args.arms.split(",")
    summary = []
    for rnd in range(args.rounds):
        for arm in (arms if rnd % 2 == 0 else arms[::-1]):
            res = run_arm(arm, rnd, args)
            summary.append({k: v for k, v in res.items() if not re.fullmatch(r"c\d+", k)}
                           | {c: {kk: vv for kk, vv in res[c].items() if kk != "rows"}
                              for c in res if re.fullmatch(r"c\d+", c)})
    scores = None if args.no_asr else score_audio()
    (OUT / "summary.json").write_text(json.dumps({"runs": summary, "asr": scores}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

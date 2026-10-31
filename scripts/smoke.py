"""MiniCPM-o 4.5 turn MRv2 Thinker, text-only greedy A/B: no speculation vs DSpark, one container.

python smoke.py [--arms off,dspark] [--max-tokens 256] [--conc 1] [--rounds 1] [--k 7] [--audio]
-> $OUT_DIR/<arm>-r<round>/server.log, results.json, metrics.txt; $OUT_DIR/summary.json

--audio asks for text + speech (English prompts only), saves each reply's WAV and
scores Whisper (openai/whisper-small.en) transcripts against the reply text.

Each arm starts `vllm serve openbmb/MiniCPM-o-4_5 --omni` with an overlay of
minicpmo_4_5_turn_mrv2.yaml written into the installed deploy dir. The dspark
arm only adds a stage-0 speculative_config. Arms alternate per round.
"""

import argparse
import base64
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
DRAFT = "Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm"
DRAFT_REV = "c4c310aca2c768343165261cb22c21b90c1f79e0"
PORT = 8091
AUDIO = False

PROMPTS = [
    "What is the capital of China? Answer in one sentence and then name three famous places there.",
    "Explain how a hash map handles collisions. Keep it under 120 words.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "List the planets of the solar system in order from the sun, one per line, with one fact each.",
    "Translate into English: 今天天气很好，我们一起去公园散步吧。",
    "请用三句话介绍一下长城。",
    "A train travels 180 km in 2 hours and then 120 km in 1.5 hours. What is its average speed? Show the steps.",
    "Write a short story of about 100 words about a robot that learns to paint.",
    "What are the main differences between TCP and UDP? Answer with a short bulleted list.",
    "Give me a recipe for scrambled eggs with numbered steps.",
    "Summarize the plot of Romeo and Juliet in four sentences.",
    "Explain the difference between supervised and unsupervised learning to a high school student.",
    "写一首关于秋天的五言绝句，并解释它的意思。",
    "What is 17 multiplied by 23? Explain how you computed it.",
    "Describe the water cycle in simple terms.",
    "Write a SQL query that returns the top 5 customers by total order amount from tables customers(id, name) and orders(id, customer_id, amount).",
]


NUM_PROMPTS = None


def prompt_set():
    prompts = [p for p in PROMPTS if p.isascii()] if AUDIO else PROMPTS
    return prompts[:NUM_PROMPTS] if NUM_PROMPTS else prompts


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
                if pcm.ndim > 1:
                    pcm = pcm.mean(axis=1)
                hyp = pipe({"raw": pcm, "sampling_rate": rate}, return_timestamps=True)["text"].strip()
                r["asr"] = hyp
                r["asr_wer_vs_text"] = wer(r.get("text"), hyp)
                # Chunked long-form decoding: a second transcript to tell ASR loops from audio.
                hyp_chunked = pipe({"raw": pcm, "sampling_rate": rate}, chunk_length_s=30, batch_size=4)["text"].strip()
                r["asr_chunked"] = hyp_chunked
                r["asr_chunked_wer_vs_text"] = wer(r.get("text"), hyp_chunked)
                r["audio_s"] = len(pcm) / rate
                scores.append({"arm": res["arm"], "round": res["round"], "conc": key, "idx": i,
                               "wer": r["asr_wer_vs_text"], "wer_chunked": r["asr_chunked_wer_vs_text"],
                               "audio_s": r["audio_s"]})
        res_file.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return scores


def deploy_dir():
    import vllm_omni

    return Path(vllm_omni.__file__).resolve().parent / "deploy"


BASE = "minicpmo_4_5_turn_mrv2.yaml"


def write_overlay(arm, k):
    lines = [f"base_config: {BASE}", "stages:", "  - stage_id: 0"]
    if arm == "dspark":
        lines += [
            "    speculative_config:",
            "      method: dspark",
            f"      model: {DRAFT}",
            f"      revision: {DRAFT_REV}",
            f"      num_speculative_tokens: {k}",
        ]
    else:
        lines += ["    enable_prefix_caching: false"]
    path = deploy_dir() / f"_smoke_{arm}.yaml"
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
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "modalities": ["text"],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 42,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if AUDIO:
        body["modalities"] = ["text", "audio"]
        body["chat_template_kwargs"]["use_tts_template"] = True
    t0 = time.perf_counter()
    r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body, timeout=600)
    dt = time.perf_counter() - t0
    row = {"prompt": prompt, "status": r.status_code, "latency_s": dt}
    try:
        data = r.json()
    except ValueError:
        row["error"] = r.text[:2000]
        return row
    if r.status_code != 200:
        row["error"] = json.dumps(data)[:2000]
        return row
    text = None
    for choice in data.get("choices", []):
        msg = choice.get("message") or {}
        if msg.get("content") and text is None:
            text = msg["content"]
            row["finish_reason"] = choice.get("finish_reason")
        audio = msg.get("audio") or {}
        if audio.get("data"):
            row["audio_b64"] = audio["data"]
    row["text"] = text
    row["id"] = data.get("id")
    row["usage"] = data.get("usage")
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
    wall = time.perf_counter() - t0
    toks = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)
    return rows, wall, toks


def run_arm(arm, rnd, args):
    out = OUT / f"{arm}-r{rnd}"
    out.mkdir(parents=True, exist_ok=True)
    overlay = write_overlay(arm, args.k)
    (out / "overlay.yaml").write_text(overlay.read_text())
    log = open(out / "server.log", "w")
    template = deploy_dir().parent / "transformers_utils" / "chat_templates" / "minicpmo45_native.jinja"
    cmd = [sys.executable, "-m", "vllm_omni.entrypoints.cli.main", "serve", MODEL, "--omni",
           "--host", "127.0.0.1", "--port", str(PORT), "--deploy-config", str(overlay),
           "--trust-remote-code", "--chat-template", str(template), "--chat-template-content-format", "openai"]
    print(f"[{arm}-r{rnd}] {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result = {"arm": arm, "round": rnd, "cmd": cmd}
    try:
        ok, startup = wait_ready(proc, args.startup_timeout)
        result["startup_s"] = startup
        result["ready"] = ok
        print(f"[{arm}-r{rnd}] ready={ok} after {startup:.0f}s", flush=True)
        if not ok:
            return result
        # Warm-up (not timed): every prompt once at the largest concurrency.
        run_batch(prompt_set()[:4], 32, max(args.conc))
        for conc in args.conc:
            rows, wall, toks = run_batch(prompt_set(), args.max_tokens, conc)
            for i, r in enumerate(rows):
                data = r.pop("audio_b64", None)
                if data:
                    path = out / f"c{conc}_{i:02d}.wav"
                    path.write_bytes(base64.b64decode(data))
                    r["audio_file"] = path.name
            result[f"c{conc}"] = {"wall_s": wall, "completion_tokens": toks,
                                  "tok_per_s": toks / wall if wall else None, "rows": rows}
            errs = [r for r in rows if r.get("error") or r.get("text") is None]
            print(f"[{arm}-r{rnd}] c{conc}: {toks} tok in {wall:.1f}s ({toks / wall:.1f} tok/s), errors={len(errs)}",
                  flush=True)
            for r in errs[:2]:
                print(f"  error: {r.get('status')} {r.get('error', '')[:500]}", flush=True)
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
        result["spec_log_lines"] = [ln for ln in text.splitlines() if re.search(r"(?i)acceptance|SpecDecoding|spec_decode|draft", ln)][-40:]
        result["error_tail"] = [ln for ln in text.splitlines() if re.search(r"(?i)error|exception|traceback", ln)][-40:]
        (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="off,dspark")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--conc", default="1")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--startup-timeout", type=int, default=2400)
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--num-prompts", type=int, default=None)
    ap.add_argument("--base", default="minicpmo_4_5_turn_mrv2.yaml", help="deploy profile the overlay inherits")
    ap.add_argument("--env", default="", help="comma-separated KEY=VALUE for the server")
    args = ap.parse_args()
    global AUDIO, NUM_PROMPTS, BASE
    AUDIO = args.audio
    BASE = args.base
    NUM_PROMPTS = args.num_prompts
    for kv in filter(None, args.env.split(",")):
        k, v = kv.split("=", 1)
        os.environ[k] = v
    args.conc = [int(c) for c in args.conc.split(",")]
    arms = args.arms.split(",")
    summary = []
    for rnd in range(args.rounds):
        for arm in (arms if rnd % 2 == 0 else arms[::-1]):
            res = run_arm(arm, rnd, args)
            summary.append({k: v for k, v in res.items() if not k.startswith("c") or k == "cmd"}
                           | {c: {kk: vv for kk, vv in res[c].items() if kk != "rows"} for c in res if re.fullmatch(r"c\d+", c)})
    # Greedy text equality across arms, per round and concurrency.
    texts = {}
    for d in sorted(OUT.glob("*-r*/results.json")):
        res = json.loads(d.read_text())
        for key, val in res.items():
            if re.fullmatch(r"c\d+", key):
                texts[(res["arm"], res["round"], key)] = [r.get("text") for r in val["rows"]]
    compare = []
    for (arm, rnd, c), t in texts.items():
        if arm == "off":
            continue
        base = texts.get(("off", rnd, c))
        if base is None:
            continue
        same = [a == b for a, b in zip(base, t)]
        compare.append({"arm": arm, "round": rnd, "conc": c, "identical": sum(same), "total": len(same),
                        "mismatch_idx": [i for i, s in enumerate(same) if not s]})
    scores = score_audio() if AUDIO else None
    (OUT / "summary.json").write_text(
        json.dumps({"runs": summary, "compare": compare, "audio_wer": scores}, indent=2, ensure_ascii=False)
    )
    print(json.dumps(compare, indent=2), flush=True)


if __name__ == "__main__":
    main()

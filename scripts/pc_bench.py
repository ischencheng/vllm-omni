"""MiniCPM-o 4.5 Thinker prefix cache A/B, one container, one server at a time.

python pc_bench.py --arms v1-off,v1-on,v2-off,v2-merge,v2-native [--rounds 1] [--workloads ...]
-> $OUT_DIR/<arm>-r<round>/{server.log,overlay.yaml,<workload>.json,<workload>.metrics.txt,rows/*.pt}

Arms: turn profile (V1) with stage-0 prefix cache off/on; turn MRv2 profile off/on
(v2-native: native KV reuse, the rows come from the ledger). -ds adds the Thinker
DSpark draft, -fp32 runs stage 0 in fp32, -tiny and -tiny2 give stage 0 a 1.6 and 0.6 GiB KV budget.
a1/a2 also had v2-merge (supports_mrv2_prefix_cache_outputs set, debug tree).
"""

import argparse
import base64
import hashlib
import io
import json
import os
import random
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
PORT = 8091

QUESTIONS = [
    "What is the capital of China? Answer in one sentence and then name three famous places there.",
    "Explain how a hash map handles collisions. Keep it under 60 words.",
    "List the planets of the solar system in order from the sun.",
    "Write a short story of about 50 words about a robot that learns to paint.",
    "What are the main differences between TCP and UDP? Answer in two sentences.",
    "Give me a recipe for scrambled eggs in three short steps.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Explain the difference between supervised and unsupervised learning in two sentences.",
    "Describe the water cycle in simple terms, in two sentences.",
    "Why is the sky blue? Answer in two sentences.",
    "Tell me a fun fact about octopuses.",
    "How do vaccines work? Keep it short.",
    "What should I pack for a weekend hiking trip?",
    "Recommend a good habit for better sleep and explain why in one sentence.",
    "What is the tallest mountain in Africa and where is it?",
    "Name two uses of copper and say why it is good for them.",
]

FOLLOW_UPS = [
    "Can you say that again in simpler words?",
    "Why does that matter? Answer briefly.",
    "Give me one concrete example.",
    "What is a common mistake people make about this?",
    "Summarize our conversation so far in one sentence.",
]

SENTENCES = [
    "The assistant answers in plain spoken English and keeps replies short enough to read aloud.",
    "Numbers are spoken in words whenever that makes them easier to follow.",
    "When a question is ambiguous the assistant picks the most common reading and says so in a few words.",
    "Safety guidance is given calmly and without alarm, with a pointer to a professional when it is needed.",
    "The store opens at nine on weekdays and at ten on weekends, and it closes one hour earlier on public holidays.",
    "Returns are accepted within thirty days with a receipt, and refunds go back to the original payment method.",
    "Loyalty members collect one point for each dollar spent and can redeem one hundred points for a five dollar voucher.",
    "Deliveries inside the city arrive within two working days, and rural addresses may take up to five working days.",
    "Fragile goods are packed in double walled boxes, and customers can ask for a photo of the packed item.",
    "The support line is open from eight in the morning until six in the evening, and chat is available all day.",
    "Bicycles sold in the store come with one free tune up, which should be booked within the first three months.",
    "Gift cards never expire and can be used online or in any branch, but they cannot be exchanged for cash.",
    "Employees may offer a price match when a local competitor shows the same item in stock at a lower price.",
    "Seasonal sales begin on the first Saturday of the month and last until the stock on the sale table runs out.",
]


def document(words, seed):
    rng = random.Random(seed)
    out, n = [], 0
    while n < words:
        s = rng.choice(SENTENCES)
        out.append(s)
        n += len(s.split())
    return " ".join(out)


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


DRAFT = "Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm"
DRAFT_REV = "c4c310aca2c768343165261cb22c21b90c1f79e0"


def write_overlay(arm):
    base = "minicpmo_4_5_turn.yaml" if arm.startswith("v1") else "minicpmo_4_5_turn_mrv2.yaml"
    enable = arm in ("v1-on", "v2-merge", "v2-native", "v2-native-ds", "v2-native-fp32", "v2-native-tiny", "v2-native-tiny2")
    text = f"base_config: {base}\nstages:\n  - stage_id: 0\n    enable_prefix_caching: {'true' if enable else 'false'}\n"
    if arm.endswith("-fp32"):
        text += "    dtype: float32\n    max_model_len: 8192\n    kv_cache_memory_bytes: 3221225472\n"
    if arm.endswith("-tiny2"):
        text += "    kv_cache_memory_bytes: 644245094\n    max_model_len: 2048\n"
    if arm.endswith("-tiny"):
        text += "    kv_cache_memory_bytes: 1717986918\n    max_model_len: 4096\n"
    if arm.endswith("-ds"):
        text += f"    speculative_config:\n      method: dspark\n      model: {DRAFT}\n      revision: {DRAFT_REV}\n      num_speculative_tokens: 7\n"
    path = deploy_dir() / f"_pc_{arm}.yaml"
    path.write_text(text)
    return path


def patch_row_dump():
    """Dump the rows llm2tts hands over: the installed tree only, for correctness runs."""
    import vllm_omni

    path = Path(vllm_omni.__file__).resolve().parent / "model_executor" / "stage_input_processors" / "minicpmo_4_5_omni.py"
    src = path.read_text()
    anchor = """                tts_token_ids_slice = torch.tensor(
                    full_token_ids[tts_bos_idx : tts_bos_idx + tts_hidden_slice.shape[0]], dtype=torch.long
                )
"""
    assert anchor in src
    hook = anchor + """                import hashlib as _h, os as _o, time as _t
                _d = _o.environ.get("SPEC_DUMP_ROWS")
                if _d:
                    torch.save({"prompt_hash": _h.sha1(str(prompt_token_ids).encode()).hexdigest(),
                                "prompt_len": len(prompt_token_ids), "ids": full_token_ids[tts_bos_idx:end_idx],
                                "rows": tts_hidden_slice.cpu(), "time": _t.time()},
                               f"{_d}/{llm_output.request_id}.pt")
"""
    path.write_text(src.replace(anchor, hook, 1))
    print("row dump patched", flush=True)


def test_audio_url(seconds=3.0, hz=220.0):
    import numpy as np
    import soundfile as sf

    t = np.arange(int(16000 * seconds)) / 16000
    pcm = (0.2 * np.sin(2 * np.pi * hz * t)).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, pcm, 16000, format="WAV")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


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


def one_request(messages, max_tokens, keep_pcm=False, modalities=("text", "audio")):
    import numpy as np
    import soundfile as sf

    body = {
        "model": MODEL,
        "messages": messages,
        "modalities": list(modalities),
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False, "use_tts_template": True},
    }
    t0 = time.perf_counter()
    row, text, pcm, rate = {}, [], [], None
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
                if chunk.get("usage"):
                    row["usage"] = chunk["usage"]
                for choice in chunk.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if not content:
                        continue
                    if chunk.get("modality") == "audio":
                        row.setdefault("ttfa_s", time.perf_counter() - t0)
                        wav, rate = sf.read(io.BytesIO(base64.b64decode(content)), dtype="float32")
                        pcm.append(wav if wav.ndim == 1 else wav.mean(axis=1))
                    else:
                        row.setdefault("ttft_s", time.perf_counter() - t0)
                        text.append(content)
    except Exception as e:  # noqa: BLE001
        row["error"] = repr(e)[:2000]
    row["latency_s"] = time.perf_counter() - t0
    row["text"] = "".join(text)
    if pcm:
        full = np.concatenate(pcm)
        row["audio_s"] = len(full) / rate
        if keep_pcm:
            row["pcm"], row["rate"] = full, rate
    return row


def run_batch(jobs, max_tokens, conc, keep_pcm=False, modalities=("text", "audio")):
    """jobs: list of message lists."""
    rows = [None] * len(jobs)
    t0 = time.perf_counter()
    sem = threading.Semaphore(max(1, conc))

    def work(i):
        with sem:
            rows[i] = one_request(jobs[i], max_tokens, keep_pcm, modalities)

    if conc <= 1:
        for i in range(len(jobs)):
            work(i)
    else:
        threads = [threading.Thread(target=work, args=(i,)) for i in range(len(jobs))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    return rows, time.perf_counter() - t0


def msgs(system, user):
    out = [{"role": "system", "content": system}] if system else []
    out.append({"role": "user", "content": [{"type": "text", "text": user}]})
    return out


def test_image_url():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 480), (200, 220, 240))
    d = ImageDraw.Draw(img)
    d.rectangle((80, 200, 300, 400), fill=(180, 60, 50))
    d.ellipse((380, 60, 520, 200), fill=(250, 210, 60))
    d.rectangle((340, 280, 600, 400), fill=(60, 130, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run_chat(conversations, turns, conc, max_tokens):
    """Each conversation keeps a thread; replies feed the next turn. Returns rows with the turn index."""
    image = test_image_url()
    results = []
    lock = threading.Lock()
    sem = threading.Semaphore(conc)

    def convo(c):
        rng = random.Random(1000 + c)
        history = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text", "text": f"Conversation {c}. Describe this picture in one sentence."},
        ]}]
        for t in range(turns):
            with sem:
                row = one_request(list(history), max_tokens)
            row["turn"], row["conversation"] = t, c
            with lock:
                results.append(row)
            history.append({"role": "assistant", "content": row.get("text") or "Okay."})
            history.append({"role": "user", "content": [{"type": "text", "text": rng.choice(FOLLOW_UPS)}]})

    t0 = time.perf_counter()
    threads = [threading.Thread(target=convo, args=(c,)) for c in range(conversations)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return results, time.perf_counter() - t0


def strip(rows):
    return [{k: v for k, v in r.items() if k not in ("pcm", "rate")} for r in rows]


def host_gpu_state():
    state = {}
    try:
        import psutil

        vm = psutil.virtual_memory()
        state["host_used_gb"] = round(vm.used / 1e9, 2)
        state["host_rss_tree_gb"] = round(
            sum(p.memory_info().rss for p in psutil.Process().children(recursive=True)) / 1e9, 2
        )
    except Exception as e:  # noqa: BLE001
        state["host_err"] = repr(e)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        state["gpu_mem_mib_util"] = out
    except Exception as e:  # noqa: BLE001
        state["gpu_err"] = repr(e)
    return state


def save_workload(out, name, rows, wall):
    (out / f"{name}.json").write_text(json.dumps({"wall_s": wall, "state": host_gpu_state(), "rows": strip(rows)}, indent=1))
    try:
        metrics = requests.get(f"http://127.0.0.1:{PORT}/metrics", timeout=5).text
        keep = [ln for ln in metrics.splitlines() if ("prefix_cache" in ln or "preemption" in ln) and not ln.startswith("#")]
        (out / f"{name}.metrics.txt").write_text("\n".join(keep) + "\n")
    except requests.RequestException as e:
        (out / f"{name}.metrics.txt").write_text(f"error {e!r}\n")
    ok = [r for r in rows if r.get("status") == 200 and not r.get("error")]
    def p(key, q=0.5):
        vals = sorted(r[key] for r in ok if key in r)
        return round(vals[int(q * (len(vals) - 1))], 3) if vals else None
    errs = [r.get("error", "")[:160] for r in rows if r.get("error")]
    print(f"  {name}: n={len(rows)} ok={len(ok)} audio={sum(1 for r in ok if r.get('audio_s'))} wall={wall:.1f}s "
          f"ttft p50={p('ttft_s')} p90={p('ttft_s', .9)} ttfa p50={p('ttfa_s')} p90={p('ttfa_s', .9)} "
          f"cached={[r.get('usage', {}).get('prompt_tokens_details') for r in ok[:2]]} "
          f"prompt={[r.get('usage', {}).get('prompt_tokens') for r in ok[:2]]} errs={errs[:2]}", flush=True)


def whisper_scores(rows):
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device=0)
    for r in rows:
        if r.get("pcm") is None:
            continue
        got = asr({"array": r["pcm"], "sampling_rate": r["rate"]})["text"]
        r["asr"], r["wer"] = got, wer(r.get("text", ""), got)


def run_arm(arm, rnd, args):
    out = OUT / f"{arm}-r{rnd}"
    (out / "rows").mkdir(parents=True, exist_ok=True)
    overlay = write_overlay(arm)
    (out / "overlay.yaml").write_text(overlay.read_text())
    log = open(out / "server.log", "w")
    template = deploy_dir().parent / "transformers_utils" / "chat_templates" / "minicpmo45_native.jinja"
    cmd = [sys.executable, "-m", "vllm_omni.entrypoints.cli.main", "serve", MODEL, "--omni",
           "--host", "127.0.0.1", "--port", str(PORT), "--deploy-config", str(overlay),
           "--trust-remote-code", "--chat-template", str(template), "--chat-template-content-format", "openai"]
    env = dict(os.environ)
    if args.dump_rows:
        env["SPEC_DUMP_ROWS"] = str(out / "rows")
    print(f"[{arm}-r{rnd}] {' '.join(cmd[2:])}", flush=True)
    t_arm = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    try:
        ok, startup = wait_ready(proc, args.startup_timeout)
        print(f"[{arm}-r{rnd}] ready={ok} after {startup:.0f}s", flush=True)
        if not ok:
            log.flush()
            lines = (out / "server.log").read_text(errors="replace").splitlines()
            for ln in lines[-60:]:
                print("  | " + ln[:300], flush=True)
            return
        system = document(args.sys_words, seed=1)
        longdoc = document(args.long_words, seed=2)
        wl = set(args.workloads.split(","))

        # Warm up JIT and graphs on every prompt size used below; each prompt starts differently.
        warm = [msgs(f"Warm up {i}. " + document(n, seed=100 + i), QUESTIONS[i]) for i, n in
                enumerate((20, 20, args.sys_words, args.sys_words, args.long_words, args.long_words))]
        run_batch(warm[:2], 32, 2)
        run_batch(warm[2:], 32, 2)
        run_chat(1, 2, 1, 16)
        print(f"  warm done {time.time() - t_arm:.0f}s {host_gpu_state()}", flush=True)

        if "correct" in wl:
            # Same prompts twice on one server. Each starts with its own text, so the first pass misses.
            jobs = [msgs(f"Note {i}. " + document(args.correct_words, seed=200 + i), QUESTIONS[i]) for i in range(8)]
            for name in ("correct_pass1", "correct_pass2"):
                os.environ.get("SPEC_DUMP_ROWS")
                rows, wall = run_batch(jobs, args.max_tokens, 1, keep_pcm=True)
                try:
                    whisper_scores(rows)
                except Exception as e:  # noqa: BLE001
                    print(f"  whisper failed: {e!r}", flush=True)
                save_workload(out, name, rows, wall)
        if "correct_mm" in wl:
            # Image and audio in the shared prefix, same prompts twice: the hit skips their encoders.
            image = test_image_url()
            jobs = []
            for i in range(6):
                content = [{"type": "text", "text": f"Item {i}. " + document(150, seed=600 + i)}]
                if i % 2 == 0:
                    content.append({"type": "image_url", "image_url": {"url": image}})
                    content.append({"type": "text", "text": "Describe the picture in one sentence."})
                else:
                    content.append({"type": "audio_url", "audio_url": {"url": test_audio_url(hz=200.0 + 40 * i)}})
                    content.append({"type": "text", "text": "Is this sound high or low? Then tell me a fun fact about sound."})
                jobs.append([{"role": "user", "content": content}])
            for name in ("correct_mm_pass1", "correct_mm_pass2"):
                rows, wall = run_batch(jobs, args.max_tokens, 1, keep_pcm=True)
                try:
                    whisper_scores(rows)
                except Exception as e:  # noqa: BLE001
                    print(f"  whisper failed: {e!r}", flush=True)
                save_workload(out, name, rows, wall)
        if "edge" in wl:
            # A reply with no Thinker rows: one token only, on a miss and on a hit, then a normal request.
            shared = document(args.correct_words, seed=800)
            first, _ = run_batch([msgs(f"Edge {rnd}. " + document(args.correct_words, seed=801), QUESTIONS[0])], 1, 1)
            seeded, _ = run_batch([msgs(shared, QUESTIONS[1])], args.max_tokens, 1)
            hit, _ = run_batch([msgs(shared, QUESTIONS[2])], 1, 1)
            after, wall = run_batch([msgs(shared, QUESTIONS[3])], args.max_tokens, 1)
            save_workload(out, "edge", first + seeded + hit + after, wall)
        if "stress" in wl:
            # Half the requests share a prefix, half do not; more KV in flight than the tiny budget holds.
            shared = document(1200, seed=700)
            jobs = []
            for i in range(48):
                head = shared if i % 2 == 0 else f"Stress {i}. " + document(1200, seed=710 + i)
                jobs.append(msgs(head, QUESTIONS[i % 16] + " Answer in about one hundred words."))
            rows, wall = run_batch(jobs, 160, 16, keep_pcm=True)
            try:
                whisper_scores(rows)
            except Exception as e:  # noqa: BLE001
                print(f"  whisper failed: {e!r}", flush=True)
            save_workload(out, "stress", rows, wall)
        if "miss_c1" in wl:
            jobs = [msgs(f"Run {rnd} miss {i}. " + document(args.sys_words, seed=300 + i), QUESTIONS[i]) for i in range(16)]
            rows, wall = run_batch(jobs, args.max_tokens, 1)
            save_workload(out, "miss_c1", rows, wall)
        if "sys_c1" in wl:
            jobs = [msgs(system, q) for q in QUESTIONS]
            rows, wall = run_batch(jobs, args.max_tokens, 1)
            save_workload(out, "sys_c1", rows, wall)
        if "sys_c8" in wl:
            jobs = [msgs(system, QUESTIONS[i % 16] + (" Be brief." if i >= 16 else "")) for i in range(32)]
            rows, wall = run_batch(jobs, args.max_tokens, 8)
            save_workload(out, "sys_c8", rows, wall)
        if "long_c1" in wl:
            jobs = [msgs(longdoc, q) for q in QUESTIONS[:8]]
            rows, wall = run_batch(jobs, args.max_tokens, 1)
            save_workload(out, "long_c1", rows, wall)
        if "chat_c4" in wl:
            rows, wall = run_chat(4, 6, 4, args.chat_max_tokens)
            save_workload(out, "chat_c4", rows, wall)
        if "text" in wl:
            # Text-only requests stream token by token, so TTFT is queueing plus prefill.
            for name, words_, seed in (("text_short", 1200, 500), ("text_long", 4000, 501)):
                doc = document(words_, seed=seed)
                jobs = [msgs(f"{name} {i}. " + document(words_, seed=seed + 1 + i), QUESTIONS[i]) for i in range(8)]
                rows, wall = run_batch(jobs, 8, 1, modalities=("text",))
                save_workload(out, f"{name}_miss", rows, wall)
                rows, wall = run_batch([msgs(doc, q) for q in QUESTIONS[:9]], 8, 1, modalities=("text",))
                save_workload(out, f"{name}_hit", rows[1:], wall)
        if "miss_c8" in wl:
            jobs = [msgs(f"Run {rnd} wave {i}. " + document(args.sys_words, seed=400 + i), QUESTIONS[i % 16]) for i in range(32)]
            rows, wall = run_batch(jobs, args.max_tokens, 8)
            save_workload(out, "miss_c8", rows, wall)
        print(f"[{arm}-r{rnd}] done in {time.time() - t_arm:.0f}s {host_gpu_state()}", flush=True)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        log.close()
        time.sleep(8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="v1-off,v1-on,v2-off,v2-merge,v2-native")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--workloads", default="correct,miss_c1,sys_c1,sys_c8,long_c1,chat_c4,text,miss_c8")
    ap.add_argument("--sys-words", type=int, default=1200)
    ap.add_argument("--long-words", type=int, default=4000)
    ap.add_argument("--correct-words", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--chat-max-tokens", type=int, default=48)
    ap.add_argument("--startup-timeout", type=int, default=1200)
    ap.add_argument("--dump-rows", action="store_true")
    args = ap.parse_args()
    if args.dump_rows:
        patch_row_dump()
    for rnd in range(args.rounds):
        for arm in args.arms.split(","):
            run_arm(arm, rnd, args)


if __name__ == "__main__":
    main()

"""MiniCPM-o 4.5 turn MRv2: Thinker DSpark with and without stage-0 prefix cache, one container.

python spec_pc.py --arms main-on:main:on:off,main-off-ds:main:off:ds,w1-on-ds:w1:on:ds \
                  [--once main-on-ds:main:on:ds] [--rounds 2]
-> $OUT_DIR/<arm>-r<round>/{server.log,overlay.yaml,results.json,rows/*.pt}; $OUT_DIR/summary.json

An arm is name:tree:cache:draft[:profile]. tree is a directory under /opt/trees
(the runner's SPEC_TREES), first on PYTHONPATH for that server; cache on/off sets
stage-0 enable_prefix_caching (anything else leaves the profile's value); draft ds
adds the Thinker DSpark draft on stage 0, as in docs/design/minicpm_o45_mrv2_performance.md;
profile is mrv2 (minicpmo_4_5_turn_mrv2.yaml, default) or turn (minicpmo_4_5_turn.yaml).
--once arms run in round 0 only. --workloads picks text, conv_audio and edge
(one-token speech replies on a miss and a hit, then a normal request).

Workloads are multi-turn conversations, one request at a time, with fixed
assistant turns in the history so every arm sends the same prompts. Each
conversation opens with its own long system text, so turn 0 misses and later
turns hit on the previous turn's prompt. text: text only (TTFT is queueing plus
prefill), conv_audio: text + audio (TTFA, Talker rows dumped by the debug tree).
SPECACC lines (debug tree, SPEC_DEBUG=1) between requests belong to that request.
"""

import argparse
import base64
import io
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
MODEL = "openbmb/MiniCPM-o-4_5"
PORT = 8091
DRAFT = "Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm"
DRAFT_REV = "c4c310aca2c768343165261cb22c21b90c1f79e0"

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

REPLIES = [
    "Sure. The short answer is that it depends on the details, but the usual case is simple and I can walk you through it.",
    "Good question. In plain words, the main idea is to keep things organized so the next step is easy to follow.",
    "Here is a quick version. First comes the basic fact, then one example, and then a short tip you can use today.",
    "Okay. I would put it this way: start with what you already know, then add one new piece at a time.",
    "In short, yes. There are a few exceptions, but for everyday use you can rely on the common rule.",
    "Let me keep it brief. The key point is the first one, and the rest mostly follows from it.",
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


def conversation(c, turns, seed_base, words):
    """Prompts for each turn of one conversation: own system text, fixed assistant turns."""
    system = f"Conversation {seed_base + c}. " + document(words, seed=seed_base + c)
    history = [{"role": "system", "content": system}]
    prompts = []
    for t in range(turns):
        history.append({"role": "user", "content": [{"type": "text", "text": QUESTIONS[(3 * c + t) % len(QUESTIONS)]}]})
        prompts.append([dict(m) for m in history])
        history.append({"role": "assistant", "content": REPLIES[(c + t) % len(REPLIES)]})
    return prompts


def request(messages, max_tokens, modalities):
    import soundfile as sf

    body = {
        "model": MODEL, "messages": messages, "modalities": list(modalities), "temperature": 0.0,
        "max_tokens": max_tokens, "seed": 42, "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False, "use_tts_template": True},
    }
    t0 = time.perf_counter()
    row, text, text_t, samples, rate = {}, [], [], 0, None
    try:
        with requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body, stream=True, timeout=900) as r:
            row["status"] = r.status_code
            if r.status_code != 200:
                row["error"] = r.text[:1000]
                return row
            for raw in r.iter_lines():
                if not raw or not raw.startswith(b"data: "):
                    continue
                if raw[6:] == b"[DONE]":
                    break
                chunk = json.loads(raw[6:])
                if chunk.get("usage"):
                    row["usage"] = chunk["usage"]
                for choice in chunk.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if not content:
                        continue
                    now = time.perf_counter() - t0
                    if chunk.get("modality") == "audio":
                        row.setdefault("ttfa_s", now)
                        wav, rate = sf.read(io.BytesIO(base64.b64decode(content)), dtype="float32")
                        samples += len(wav)
                    else:
                        row.setdefault("ttft_s", now)
                        text.append(content)
                        text_t.append(now)
    except Exception as e:  # noqa: BLE001
        row["error"] = repr(e)[:1000]
    row["latency_s"] = time.perf_counter() - t0
    row["text"] = "".join(text)
    row["text_chunks"] = len(text_t)
    if len(text_t) >= 2:
        row["text_span_s"] = text_t[-1] - text_t[0]
    if samples and rate:
        row["audio_s"] = samples / rate
    return row


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


def overlay(tree_dir, cache, draft, profile):
    base = {"mrv2": "minicpmo_4_5_turn_mrv2.yaml", "turn": "minicpmo_4_5_turn.yaml"}[profile]
    lines = [f"base_config: {base}", "stages:", "  - stage_id: 0"]
    if cache in ("on", "off"):
        lines += [f"    enable_prefix_caching: {'true' if cache == 'on' else 'false'}"]
    if draft == "ds":
        lines += ["    speculative_config:", "      method: dspark", f"      model: {DRAFT}",
                  f"      revision: {DRAFT_REV}", "      num_speculative_tokens: 7"]
    target = tree_dir / "vllm_omni" / "deploy" / f"_sp_{profile}_{cache}_{draft}.yaml"
    target.write_text("\n".join(lines) + "\n")
    return target


def specacc(log_path):
    out = []
    for line in log_path.read_text(errors="replace").splitlines():
        m = re.search(r"SPECACC nreq=(\d+) ntok=(\d+) ndraft=(\d+) nsamp=\[([^\]]*)\]", line)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), [int(x) for x in m.group(4).split(",") if x.strip()]))
    return out


def run_workload(name, convs, max_tokens, modalities, log_path):
    rows = []
    for c, prompts in enumerate(convs):
        for t, messages in enumerate(prompts):
            before = len(specacc(log_path))
            r = request(messages, max_tokens, modalities)
            time.sleep(0.3)
            steps = specacc(log_path)[before:]
            verify = [s for s in steps if s[2] > 0]
            r.update(workload=name, conversation=c, turn=t, verify_steps=len(verify),
                     verify_tokens=sum(sum(s[3]) for s in verify), steps=len(steps))
            r["cached_tokens"] = ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
            r["prompt_tokens"] = (r.get("usage") or {}).get("prompt_tokens")
            r["completion_tokens"] = (r.get("usage") or {}).get("completion_tokens")
            rows.append(r)
    return rows


def describe(rows):
    out = {}
    for label, sel in (("miss", [r for r in rows if r["turn"] == 0]), ("hit", [r for r in rows if r["turn"] > 0])):
        ok = [r for r in sel if r.get("status") == 200 and not r.get("error")]
        def mean(key):
            vals = [r[key] for r in ok if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None
        vs = sum(r["verify_steps"] for r in ok)
        out[label] = {
            "n": len(sel), "ok": len(ok), "ttft_s": mean("ttft_s"), "ttfa_s": mean("ttfa_s"), "latency_s": mean("latency_s"),
            "audio_s": mean("audio_s"), "cached_tokens": mean("cached_tokens"), "prompt_tokens": mean("prompt_tokens"),
            "completion_tokens": mean("completion_tokens"),
            "tokens_per_verify": round(sum(r["verify_tokens"] for r in ok) / vs, 3) if vs else None,
            "noaudio": sum(1 for r in ok if "audio" in r.get("modalities", []) and not r.get("audio_s")),
        }
    return out


def run_arm(spec, rnd, args):
    name, tree, cache, draft, *rest = spec.split(":")
    profile = rest[0] if rest else "mrv2"
    out = OUT / f"{name}-r{rnd}"
    (out / "rows").mkdir(parents=True, exist_ok=True)
    tree_dir = Path("/opt/trees") / tree
    ov = overlay(tree_dir, cache, draft, profile)
    (out / "overlay.yaml").write_text(ov.read_text())
    template = tree_dir / "vllm_omni" / "transformers_utils" / "chat_templates" / "minicpmo45_native.jinja"
    cmd = [sys.executable, "-m", "vllm_omni.entrypoints.cli.main", "serve", MODEL, "--omni",
           "--host", "127.0.0.1", "--port", str(PORT), "--deploy-config", str(ov),
           "--trust-remote-code", "--chat-template", str(template), "--chat-template-content-format", "openai"]
    env = dict(os.environ, PYTHONPATH=str(tree_dir), SPEC_DEBUG="1", SPEC_DUMP_ROWS=str(out / "rows"))
    log_path = out / "server.log"
    log = open(log_path, "w")
    print(f"[{name}-r{rnd}] tree={tree} cache={cache} draft={draft}", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env, cwd="/work")
    result = {"arm": name, "tree": tree, "cache": cache, "draft": draft, "profile": profile, "round": rnd}
    try:
        ok, startup = wait_ready(proc, args.startup_timeout)
        result.update(ready=ok, startup_s=round(startup))
        print(f"[{name}-r{rnd}] ready={ok} after {startup:.0f}s", flush=True)
        if not ok:
            log.flush()
            lines = log_path.read_text(errors="replace").splitlines()
            result["startup_errors"] = [ln[-500:] for ln in lines if re.search(r"Error|refused|not supported", ln)][:12]
            for ln in result["startup_errors"][:6]:
                print("  | " + ln, flush=True)
            return result
        result["vllm_omni_file"] = subprocess.run(
            [sys.executable, "-c", "import vllm_omni; print(vllm_omni.__file__)"], env=env, capture_output=True, text=True
        ).stdout.strip()
        # Warm up graphs and JIT on both workloads, with their own system texts.
        run_workload("warm_text", [conversation(c, 3, 900, args.words) for c in range(2)], 32, ("text",), log_path)
        run_workload("warm_audio", [conversation(c, 2, 950, args.words) for c in range(1)], 32, ("text", "audio"), log_path)
        rows = []
        wanted = set(args.workloads.split(","))
        if "edge" in wanted:
            # One-token speech replies: the Thinker hands over no reply rows.
            shared = conversation(0, 3, 300, args.words)
            edge = []
            for label, messages, max_tokens in (("miss_1tok", conversation(1, 1, 310, args.words)[0], 1),
                                                 ("seed", shared[0], args.audio_max_tokens),
                                                 ("hit_1tok", shared[1], 1), ("after", shared[2], args.audio_max_tokens)):
                r = request(messages, max_tokens, ("text", "audio"))
                r.update(workload="edge", label=label)
                edge.append(r)
                print(f"[{name}-r{rnd}] edge {label}: status={r.get('status')} audio_s={r.get('audio_s')} "
                      f"cached={((r.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens')} "
                      f"err={(r.get('error') or '')[:120]}", flush=True)
            rows += edge
        for wl, seed, convs, turns, max_tokens, modalities in (
            ("text", 100, args.text_convs, args.turns, args.text_max_tokens, ("text",)),
            ("conv_audio", 200, args.audio_convs, args.audio_turns, args.audio_max_tokens, ("text", "audio")),
        ):
            if wl not in wanted:
                continue
            wl_rows = run_workload(wl, [conversation(c, turns, seed, args.words) for c in range(convs)], max_tokens, modalities, log_path)
            for r in wl_rows:
                r["modalities"] = list(modalities)
            rows += wl_rows
            result[wl] = describe(wl_rows)
            print(f"[{name}-r{rnd}] {wl}: {json.dumps(result[wl])}", flush=True)
        result["rows"] = rows
        try:
            metrics = requests.get(f"http://127.0.0.1:{PORT}/metrics", timeout=10).text
            result["metrics"] = [ln for ln in metrics.splitlines()
                                 if "prefix_cache" in ln and not ln.startswith("#")]
        except requests.RequestException as e:
            result["metrics"] = [f"error {e!r}"]
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
        time.sleep(5)
        text = log_path.read_text(errors="replace")
        result["error_tail"] = [ln[-400:] for ln in text.splitlines() if re.search(r"(?i)error|exception|traceback|rows stop", ln)][-30:]
        (out / "results.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--once", default="")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--words", type=int, default=900)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--text-convs", type=int, default=4)
    ap.add_argument("--text-max-tokens", type=int, default=128)
    ap.add_argument("--audio-convs", type=int, default=3)
    ap.add_argument("--audio-turns", type=int, default=5)
    ap.add_argument("--audio-max-tokens", type=int, default=64)
    ap.add_argument("--startup-timeout", type=int, default=2400)
    ap.add_argument("--workloads", default="text,conv_audio")
    args = ap.parse_args()
    arms = args.arms.split(",")
    summary = []
    for spec in filter(None, args.once.split(",")):
        res = run_arm(spec, 0, args)
        summary.append({k: v for k, v in res.items() if k != "rows"})
    for rnd in range(args.rounds):
        for spec in (arms if rnd % 2 == 0 else arms[::-1]):
            res = run_arm(spec, rnd, args)
            summary.append({k: v for k, v in res.items() if k != "rows"})
            (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()

"""JoyAI duplex: when does the first audio of a spoken observation reach the client?

Arm A: upstream JoyAI duplex deploy (all-sync JoyAI -> Qwen3-TTS -> Code2Wav).
  One session, a warm-up observation, then several image observations. Per response:
  commit -> response.created, first transcript event, first audio delta, response.done,
  audio delta count and audio seconds.
Arm B: Qwen3-TTS alone with its own streaming deploy (async-chunk talker -> Code2Wav),
  /v1/audio/speech stream_format=audio, the same texts JoyAI spoke in arm A, same
  speaker. Per text: request -> first PCM byte, last byte. Runs with the deploy's
  default runner (v2) and with model_runner v1 (the runner JoyAI's TTS stages use).

Arm C: the streaming overlay (joyai_vl_interaction_duplex_streaming.yaml) with the same
  driver as arm A; it needs a tree that has the overlay.

Everything runs in one container on one GPU, one server at a time.

  VLLM_OMNI_TREE=<tree> OUT_DIR=<out> python measure.py --arms a,c
"""

import asyncio
import base64
import io
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
# The vllm-omni tree that is installed (pip install .); its deploy/ YAMLs are used.
REPO = Path(os.environ.get("VLLM_OMNI_TREE", "/opt/vllm-omni"))
JOYAI = "jdopensource/JoyAI-VL-Interaction-Preview"
TTS = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
INSTRUCTIONS = "Whenever a new image arrives, describe what you see in two or three spoken sentences."
QUESTION = "Please describe each new image aloud in two or three sentences."


def log(*args):
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def start_server(model, deploy, port, name):
    cmd = ["vllm", "serve", model, "--omni", "--deploy-config", str(deploy), "--port", str(port)]
    log("start", name, " ".join(cmd))
    out = open(OUT / f"server_{name}.log", "w")
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, cwd=str(REPO))
    deadline = time.time() + 3600
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    log("ready", name)
                    return proc
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"{name} server not ready")


def stop_server(proc, name):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=90)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
    log("stopped", name)
    time.sleep(10)


def images():
    from PIL import Image, ImageDraw

    out = []
    try:
        from vllm.assets.image import ImageAsset

        for name in ("stop_sign", "cherry_blossom"):
            out.append((name, ImageAsset(name).pil_image.convert("RGB")))
    except Exception as exc:  # network or asset change: drawn images still run
        log("image assets unavailable:", exc)
    scene = Image.new("RGB", (448, 336), (135, 206, 235))
    d = ImageDraw.Draw(scene)
    d.rectangle([0, 230, 448, 336], fill=(60, 160, 60))
    d.ellipse([340, 30, 410, 100], fill=(255, 215, 0))
    d.rectangle([120, 140, 260, 250], fill=(200, 40, 40))
    d.polygon([(110, 140), (190, 80), (270, 140)], fill=(120, 60, 30))
    d.rectangle([175, 195, 205, 250], fill=(90, 50, 20))
    out.append(("drawn_house", scene))
    board = Image.new("RGB", (448, 336), (245, 245, 245))
    d = ImageDraw.Draw(board)
    d.rectangle([20, 20, 428, 316], outline=(30, 30, 30), width=4)
    for i, line in enumerate(("TEAM MEETING", "Thursday 3 PM", "Room 204", "Bring the Q3 report")):
        d.text((50, 60 + 50 * i), line, fill=(20, 20, 160))
    out.append(("drawn_whiteboard", board))
    warm = Image.new("RGB", (224, 224), "red")
    return ("warmup_red", warm), out


def jpeg_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


async def arm_a(port, label):
    from vllm_omni.clients.duplex import DuplexClient, EventCollector, SessionConfig

    warm, imgs = images()
    config = SessionConfig(
        auto_response=False,
        temperature=0.0,
        instructions=INSTRUCTIONS,
        extra_body={"tts_language": "English"},
    )
    collector = EventCollector()
    rows = []
    url = f"ws://127.0.0.1:{port}/v1/realtime?duplex=1"
    async with DuplexClient(url, model=JOYAI, config=config, reconnect=None) as client:
        consumer = asyncio.create_task(collector.consume(client))
        await asyncio.sleep(0)
        await client.send(
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": QUESTION}]},
            }
        )
        pending = None
        for name, img in [warm, *imgs]:
            done = asyncio.create_task(client.wait_for("response.done", timeout_s=300))
            await asyncio.sleep(0)
            await client.append_audio(b"", video_frames=[jpeg_b64(img)])
            t_commit = time.monotonic()
            await client.commit(create_response=True)
            if pending is not None:
                await client.ack_playback(pending[0], response_id=pending[1])
            terminal = await done
            rid = terminal.response_id
            ev = [(e, t) for e, t in zip(collector.events, collector.event_received_at_s) if t >= t_commit]
            mine = [(e, t) for e, t in ev if collector.response_id(e) == rid]

            def first(pred):
                for e, t in mine:
                    if pred(e.get("type", "")):
                        return round((t - t_commit) * 1000, 1)
                return None

            pcm = collector.audio_bytes(rid)
            sr = collector.output_sample_rate_hz
            audio_s = len(pcm) / 2 / sr
            text = collector.response_text(rid)
            row = {
                "image": name,
                "response_id": rid,
                "status": terminal.raw.get("response", {}).get("status"),
                "created_ms": first(lambda t: t == "response.created"),
                "first_transcript_ms": first(lambda t: "transcript" in t or t.endswith("text.delta")),
                "first_audio_ms": first(lambda t: t == "response.output_audio.delta"),
                "done_ms": round((collector.last_received_at("response.done") - t_commit) * 1000, 1),
                "audio_delta_events": sum(e.get("type") == "response.output_audio.delta" for e, _ in mine),
                "audio_s": round(audio_s, 3),
                "text": text,
                "event_types": [e.get("type") for e, _ in mine],
            }
            rows.append(row)
            log(label, json.dumps({k: v for k, v in row.items() if k != "event_types"}))
            pending = (audio_s * 1000, rid) if pcm else None
        if pending is not None:
            await client.ack_playback(pending[0], response_id=pending[1])
        await client.close()
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
    (OUT / f"events_{label}.json").write_text(json.dumps(collector.events, indent=1, default=str))
    return rows


async def silence_probe(port, label):
    """A session told to stay silent on ordinary images: silent observations must finish promptly."""
    from vllm_omni.clients.duplex import DuplexClient, EventCollector, SessionConfig

    _, imgs = images()
    config = SessionConfig(
        auto_response=False,
        temperature=0.0,
        instructions="Only speak if you see fire or smoke. Otherwise stay silent.",
        extra_body={"tts_language": "English"},
    )
    collector = EventCollector()
    rows = []
    url = f"ws://127.0.0.1:{port}/v1/realtime?duplex=1"
    async with DuplexClient(url, model=JOYAI, config=config, reconnect=None) as client:
        consumer = asyncio.create_task(collector.consume(client))
        await asyncio.sleep(0)
        for name, img in [i for i in imgs if i[0] in ("drawn_house", "cherry_blossom")]:
            done = asyncio.create_task(client.wait_for("response.done", timeout_s=120))
            await asyncio.sleep(0)
            await client.append_audio(b"", video_frames=[jpeg_b64(img)])
            t_commit = time.monotonic()
            await client.commit(create_response=True)
            try:
                terminal = await done
            except Exception as exc:
                rows.append({"image": name, "error": repr(exc), "waited_ms": round((time.monotonic() - t_commit) * 1000, 1)})
                log(label, "silence", json.dumps(rows[-1]))
                break
            rid = terminal.response_id
            pcm = collector.audio_bytes(rid)
            row = {
                "image": name,
                "status": terminal.raw.get("response", {}).get("status"),
                "done_ms": round((collector.last_received_at("response.done") - t_commit) * 1000, 1),
                "audio_s": round(len(pcm) / 2 / collector.output_sample_rate_hz, 3),
                "text": collector.response_text(rid),
            }
            rows.append(row)
            log(label, "silence", json.dumps(row))
            if pcm:
                await client.ack_playback(len(pcm) * 500 / collector.output_sample_rate_hz, response_id=rid)
        await client.close()
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
    (OUT / f"events_{label}_silence.json").write_text(json.dumps(collector.events, indent=1, default=str))
    return rows


def speech_stream(port, text):
    body = json.dumps(
        {
            "model": TTS,
            "input": text,
            "voice": "Vivian",
            "language": "English",
            "task_type": "CustomVoice",
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/audio/speech", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    first = None
    total = 0
    chunks = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        while True:
            b = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not b:
                break
            if first is None:
                first = time.monotonic()
            total += len(b)
            chunks += 1
    t1 = time.monotonic()
    return {
        "first_byte_ms": None if first is None else round((first - t0) * 1000, 1),
        "last_byte_ms": round((t1 - t0) * 1000, 1),
        "reads": chunks,
        "audio_s": round(total / 2 / 24000, 3),
    }


def arm_b(port, texts, name):
    rows = []
    speech_stream(port, "Warm up the speech server once before measuring.")
    for i, text in enumerate(texts):
        for rep in range(2):
            row = {"arm": name, "text_index": i, "rep": rep, **speech_stream(port, text)}
            rows.append(row)
            log("B", json.dumps(row))
    return rows


def main():
    import argparse

    ap = argparse.ArgumentParser()
    # a: upstream duplex deploy (all-sync); c: streaming overlay (prototype tree); b: Qwen3-TTS alone.
    ap.add_argument("--arms", default="a,b")
    args = ap.parse_args()
    arms = args.arms.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    deploy_dir = REPO / "vllm_omni" / "deploy"
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], capture_output=True, text=True)
    (OUT / "gpu.txt").write_text(gpu.stdout)
    summary = {"gpu": gpu.stdout.strip(), "arms": arms}
    joyai_arms = {
        "a": ("joyai_duplex", "joyai_vl_interaction_duplex.yaml"),
        "c": ("joyai_duplex_streaming", "joyai_vl_interaction_duplex_streaming.yaml"),
        "r": ("joyai_duplex_streaming_ramp", "joyai_streaming_ramp_measure.yaml"),
        "i4": ("joyai_duplex_streaming_ic4", "joyai_streaming_ic4_measure.yaml"),
    }
    # Chunk-size variants of the streaming overlay (written here, not shipped).
    for fname, extra in (
        ("joyai_streaming_ramp_measure.yaml", "      codec_chunk_ramp: [4, 4, 8, 16, 25]\n"),
        ("joyai_streaming_ic4_measure.yaml", "      initial_codec_chunk_frames: 4\n"),
    ):
        (deploy_dir / fname).write_text(
            "base_config: joyai_vl_interaction_duplex_streaming.yaml\n"
            "connectors:\n  connector_of_shared_memory:\n    extra:\n" + extra
        )
    for arm in arms:
        if arm not in joyai_arms:
            continue
        name, deploy = joyai_arms[arm]
        proc = start_server(JOYAI, deploy_dir / deploy, 8099, name)
        try:
            summary[f"arm_{arm}"] = asyncio.run(arm_a(8099, name))
            summary[f"arm_{arm}_silence"] = asyncio.run(silence_probe(8099, name))
        finally:
            stop_server(proc, name)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    if "b" in arms:
        texts = [r["text"] for r in summary.get("arm_a", [])[1:] if r["text"] and r["audio_s"] > 0]
        summary["arm_b_texts"] = texts
        v1 = deploy_dir / "qwen3_tts_v1_measure.yaml"
        v1.write_text(f"base_config: {deploy_dir / 'qwen3_tts.yaml'}\nmodel_runner: v1\n")
        summary["arm_b"] = []
        for name, deploy in (("tts_stream_v1", v1), ("tts_stream_default", deploy_dir / "qwen3_tts.yaml")):
            proc = start_server(TTS, deploy, 8091, name)
            try:
                summary["arm_b"] += arm_b(8091, texts, name)
            finally:
                stop_server(proc, name)
            (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    log("done")


if __name__ == "__main__":
    sys.exit(main())

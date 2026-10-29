"""Whisper transcripts of recorded replies (default vs streaming deploy) against JoyAI's text.

python asr.py  ->  $OUT_DIR/asr.json
Inputs: asr_input/*.wav + manifest.json next to this script, one WAV per reply,
rebuilt from the response.output_audio.delta events of a run (events_*.json).
"""

import json
import os
import re
import wave
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
IN = Path(__file__).resolve().parent / "asr_input"


def words(text):
    return re.sub(r"[^a-z0-9' ]", " ", text.lower().replace("-", " ")).split()


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


def main():
    from transformers import pipeline

    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device="cpu")
    rows = []
    for m in json.load(open(IN / "manifest.json")):
        with wave.open(str(IN / m["file"])) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
            rate = w.getframerate()
        hyp = pipe({"raw": pcm, "sampling_rate": rate}, return_timestamps=True)["text"].strip()
        frame = rate // 10
        rms = [float(np.sqrt(np.mean(pcm[i : i + frame] ** 2))) for i in range(0, len(pcm) - frame + 1, frame)]
        voiced = [i for i, v in enumerate(rms) if v > 0.01]
        row = {
            **m,
            "asr": hyp,
            "wer": round(wer(m["text"], hyp), 3),
            "first_voiced_s": None if not voiced else voiced[0] / 10,
            "last_voiced_s": None if not voiced else (voiced[-1] + 1) / 10,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    (OUT / "asr.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

# Ovi 1.1 — Audio + Video Joint Generation

[Ovi 1.1](https://github.com/character-ai/Ovi) is Character.AI's twin-backbone
DiT for synchronized text/image-to-video-and-audio generation. Each diffusion
step runs a video DiT and an audio DiT in lockstep with bidirectional
cross-modal fusion, producing time-aligned 5-second or 10-second clips at
720×720 or 960×960 with 24 FPS / 16 kHz audio.

## Setup

Three Hugging Face repos are required (the script pulls them on first run):

| Repo | Purpose |
| --- | --- |
| `chetwinlow1/Ovi` | Fusion DiT checkpoint (passed as `--model`) |
| `Wan-AI/Wan2.2-TI2V-5B` | UMT5 text encoder + WAN VAE 2.2 (video) |
| `hkchengrex/MMAudio` | MMAudio TOD VAE + BigVGAN vocoder (audio) |

The Ovi repo only ships a `model.safetensors` plus a near-empty
`config.json`. After download, append the architecture name so vLLM-Omni can
resolve the pipeline class:

```bash
huggingface-cli download chetwinlow1/Ovi --local-dir ./ckpts/Ovi
python - <<'PY'
import json, pathlib
p = pathlib.Path("ckpts/Ovi/config.json")
cfg = json.loads(p.read_text())
cfg["architectures"] = ["Ovi11Pipeline"]
p.write_text(json.dumps(cfg, indent=2))
PY
```

## Variants

| Variant | Checkpoint filename | Resolution | Duration |
| --- | --- | --- | --- |
| `720x720_5s` | `model.safetensors` | 720×720 | 5 s |
| `960x960_5s` (default) | `model_960x960.safetensors` | 960×960 | 5 s |
| `960x960_10s` | `model_960x960_10s.safetensors` | 960×960 | 10 s |

Selection is via `tf_model_config={"variant": "..."}` on the `Omni` constructor.

## Run

```bash
python ovi_1_1_t2v.py
```

## Audio prompts

The 720×720 5s variant expects audio inside `<AUDCAP>...<ENDAUDCAP>` tags
while the 960×960 variants expect plain `Audio: ...`. The pipeline
auto-converts between the two formats per variant, so you can write either
style — the formatter applies before tokenization.

## Outputs

Each request returns a single `OmniRequestOutput` whose
`multimodal_output` payload is:

```python
{
    "video": np.ndarray,       # [B, F, H, W, 3] uint8-friendly
    "audio": np.ndarray,       # 1-D waveform
    "audio_sample_rate": 16000,
    "fps": 24.0,
}
```

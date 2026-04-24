# Ovi 1.1

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/ovi_1_1>.

[Ovi 1.1](https://github.com/character-ai/Ovi) is Character.AI's twin-backbone
DiT for synchronized text/image-to-video-and-audio generation. Each diffusion
step runs a video DiT and an audio DiT in lockstep with bidirectional
cross-modal fusion, producing time-aligned 5 s or 10 s clips at 720×720 or
960×960 with 24 fps + 16 kHz audio.

## Setup

Three Hugging Face repos are required (the example pulls them on first run):

| Repo | Purpose |
| --- | --- |
| `chetwinlow1/Ovi` | Fusion DiT checkpoint (passed as `--model`) |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | UMT5 text encoder + WAN VAE 2.2 (video) |
| `hkchengrex/MMAudio` | MMAudio TOD VAE + BigVGAN vocoder (audio) |

The Ovi repo only ships `model*.safetensors` files plus a nearly empty
`config.json`. Patch it with `architectures` (for pipeline class routing)
and `model_type` (for the stage-config resolver) before launching:

```bash
huggingface-cli download chetwinlow1/Ovi --local-dir ./ckpts/Ovi
python - <<'PY'
import json, pathlib
p = pathlib.Path("ckpts/Ovi/config.json")
cfg = json.loads(p.read_text())
cfg["architectures"] = ["Ovi11Pipeline"]
cfg["model_type"] = "ovi_1_1"
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

### T2V (text → video + audio)

```bash
python ovi_1_1_t2v.py
```

### I2V (image + text → video + audio)

Takes a reference image as the clean first frame. Output resolution is
derived from the image's aspect ratio and the variant's target area —
don't pass `height` / `width` for I2V.

```bash
python ovi_1_1_i2v.py \
  --model ./ckpts/Ovi \
  --image ./first_frame.png \
  --prompt "... Audio: ..." \
  --output ovi_1_1_i2v.mp4
```

## Audio prompts

The 720×720 5s variant expects audio inside `<AUDCAP>...<ENDAUDCAP>` tags
while the 960×960 variants expect plain `Audio: ...`. The pipeline
auto-converts between the two formats per variant, so you can write either
style — the formatter applies before tokenization.

## Outputs

Each request returns a single `OmniRequestOutput`. The diffusion engine
routes decoded video frames into the `images` attribute and the paired
audio into `multimodal_output`:

```python
ro = outputs[0]
video = ro.images[0]                                       # np.ndarray [B, F, H, W, 3] float32 in [0, 1]
audio = ro.multimodal_output["audio"]                      # 1-D np.ndarray, 16 kHz
sample_rate = ro.multimodal_output["audio_sample_rate"]    # 16000
fps = ro.multimodal_output["fps"]                          # 24.0
```

## Performance

End-to-end on a single H200 80 GB (variant `960x960_5s`, 50 steps, seed
42, `BF16`):

| Stage | Time | Memory |
| --- | --- | --- |
| Model load | 23.7 s | 34.2 GiB |
| Engine init (incl. UMT5, VAEs) | 68 s | — |
| Generation | 146 s (2.92 s / step) | 52 GB allocated / 59 GB reserved |
| Output | 5.02 s mp4 — H.264 960×960 24 fps + AAC 16 kHz mono | — |

Numerically aligned with upstream Ovi inference (`character-ai/Ovi`)
on the same prompt + seed: temporal frame variance 24.8 vs 25.3 (within
2 %), audio dynamic range ±0.026 vs ±0.032.

## Example materials

??? abstract "ovi_1_1_t2v.py"
    ``````py
    --8<-- "examples/offline_inference/ovi_1_1/ovi_1_1_t2v.py"
    ``````

??? abstract "ovi_1_1_i2v.py"
    ``````py
    --8<-- "examples/offline_inference/ovi_1_1/ovi_1_1_i2v.py"
    ``````

# SPDX-License-Identifier: Apache-2.0
"""Offline T2V inference with Ovi 1.1.

Generates a 5-second 960x960 video with synchronized audio.

Prerequisites
-------------
The following Hugging Face repos are pulled lazily on first run:

  * ``chetwinlow1/Ovi``                — fusion DiT (this is ``--model``)
  * ``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` — UMT5 text encoder + WAN VAE 2.2
  * ``hkchengrex/MMAudio``             — MMAudio TOD VAE + BigVGAN vocoder

Ovi's repo only contains ``model*.safetensors`` files plus a nearly empty
``config.json``. Patch the config so vLLM-Omni can resolve the pipeline:

.. code-block:: bash

    huggingface-cli download chetwinlow1/Ovi --local-dir ./ckpts/Ovi
    python - <<'PY'
    import json, pathlib
    p = pathlib.Path("ckpts/Ovi/config.json")
    cfg = json.loads(p.read_text())
    cfg["architectures"] = ["Ovi11Pipeline"]
    cfg["model_type"] = "ovi_1_1"
    p.write_text(json.dumps(cfg, indent=2))
    PY
"""

import imageio
import numpy as np

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main():
    omni = Omni(
        model="./ckpts/Ovi",
        # `variant` selects which of the three model_*.safetensors to load.
        # Choices: "720x720_5s", "960x960_5s" (default), "960x960_10s".
        tf_model_config={"variant": "960x960_5s"},
    )

    prompt = (
        "A golden retriever puppy bounding through a sunlit meadow."
        " Audio: gentle wind, distant birdsong, occasional happy panting."
    )

    outputs = omni.generate(
        prompt,
        OmniDiffusionSamplingParams(
            height=960,
            width=960,
            num_inference_steps=50,
            seed=42,
            extra_args={
                "video_guidance_scale": 4.0,
                "audio_guidance_scale": 3.0,
                "slg_layer": 11,
                "shift": 5.0,
            },
        ),
    )

    # The diffusion engine puts decoded video frames in `ro.images` (shape
    # [B, F, H, W, C], float32 in [0, 1]) and the paired audio payload in
    # `ro.multimodal_output` alongside `audio_sample_rate` and `fps`.
    ro = outputs[0]
    video = np.asarray(ro.images[0])
    audio = np.asarray(ro.multimodal_output["audio"]).flatten()
    fps = float(ro.multimodal_output.get("fps", 24))
    sample_rate = int(ro.multimodal_output.get("audio_sample_rate", 16000))

    out_path = "ovi_1_1_t2v.mp4"
    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        audio_data=audio,
    )
    for frame in video[0]:
        writer.append_data((frame * 255).clip(0, 255).astype(np.uint8))
    writer.close()
    print(f"Saved {out_path} ({video.shape[1]} frames @ {fps} fps, audio {audio.shape[0]}/{sample_rate}Hz)")


if __name__ == "__main__":
    main()

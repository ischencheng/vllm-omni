# SPDX-License-Identifier: Apache-2.0
"""Offline T2V inference with Ovi 1.1.

Generates a 5-second 960x960 video with synchronized audio.

Prerequisites
-------------
The following Hugging Face repos are pulled lazily on first run:

  * ``chetwinlow1/Ovi``                — fusion DiT (this is ``--model``)
  * ``Wan-AI/Wan2.2-TI2V-5B``          — UMT5 text encoder + WAN VAE 2.2
  * ``hkchengrex/MMAudio``             — MMAudio TOD VAE + BigVGAN vocoder

Ovi's repo only contains a ``model.safetensors`` file plus an essentially
empty ``config.json``. Append the architecture name so vLLM-Omni can route
to the right pipeline class:

.. code-block:: bash

    huggingface-cli download chetwinlow1/Ovi --local-dir ./ckpts/Ovi
    python - <<'PY'
    import json, pathlib
    p = pathlib.Path("ckpts/Ovi/config.json")
    cfg = json.loads(p.read_text())
    cfg["architectures"] = ["Ovi11Pipeline"]
    p.write_text(json.dumps(cfg, indent=2))
    PY
"""

from vllm_omni.entrypoints.omni import Omni


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
        sampling_params={
            "height": 960,
            "width": 960,
            "num_inference_steps": 50,
            "seed": 42,
            "extra_args": {
                "video_guidance_scale": 4.0,
                "audio_guidance_scale": 3.0,
                "slg_layer": 11,
                "shift": 5.0,
            },
        },
    )

    request_output = outputs[0].request_output
    multimodal = request_output.outputs[0].multimodal_output
    video = multimodal["video"]
    audio = multimodal["audio"]
    fps = multimodal.get("fps", 24)
    sample_rate = multimodal.get("audio_sample_rate", 16000)

    # Save mp4 with embedded audio. Requires `pip install imageio[ffmpeg]`.
    import imageio
    import numpy as np

    out_path = "ovi_1_1_t2v.mp4"
    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        audio_fps=sample_rate,
        audio_data=np.asarray(audio).flatten(),
    )
    for frame in video[0]:
        writer.append_data((frame * 255).astype("uint8"))
    writer.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

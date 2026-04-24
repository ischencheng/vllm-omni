# SPDX-License-Identifier: Apache-2.0
"""Offline I2V inference with Ovi 1.1.

Takes a single input image + text prompt and generates a 5-second video
with synchronized audio. The pipeline treats the input image as the
clean first frame and animates the rest of the sequence.

Prerequisites — see ``ovi_1_1_t2v.py`` for the three HF repos that are
pulled on first run and the config.json patch step.

Output resolution is derived automatically from the input image's
aspect ratio and the variant's target area (960×960 = 921 600 pixels
for the ``960x960_5s`` variant). For a 1424×736 input this produces a
704×1344 video, matching upstream Ovi's ``preprocess_image_tensor`` +
``snap_hw_to_multiple_of_32`` behavior.
"""

import argparse

import numpy as np
from PIL import Image

from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./ckpts/Ovi")
    parser.add_argument("--image", required=True, help="Path to input PNG/JPG.")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Describe the motion the model should animate + the audio content (e.g. 'Audio: ambient hum').",
    )
    parser.add_argument("--output", default="ovi_1_1_i2v.mp4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    args = parser.parse_args()

    omni = Omni(
        model=args.model,
        # Choose the variant that matches the clip length you want.
        tf_model_config={"variant": "960x960_5s"},
    )

    image = Image.open(args.image).convert("RGB")

    # Do NOT pass height/width: the pre-process derives the aspect-preserving
    # output shape from the image automatically.
    sp = OmniDiffusionSamplingParams(
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        extra_args={
            "video_guidance_scale": 4.0,
            "audio_guidance_scale": 3.0,
            "slg_layer": 11,
            "shift": 5.0,
        },
    )

    outputs = omni.generate(
        {"prompt": args.prompt, "multi_modal_data": {"image": image}},
        sp,
    )

    ro = outputs[0]
    video = np.asarray(ro.images[0])
    audio = np.asarray(ro.multimodal_output["audio"]).flatten()
    fps = float(ro.multimodal_output["fps"])
    sr = int(ro.multimodal_output["audio_sample_rate"])

    frames = (video[0] * 255).clip(0, 255).astype(np.uint8)
    mp4_bytes = mux_video_audio_bytes(
        frames,
        audio_waveform=audio,
        fps=fps,
        audio_sample_rate=sr,
    )
    with open(args.output, "wb") as f:
        f.write(mp4_bytes)
    print(
        f"Saved {args.output} ({video.shape[1]} frames @ {fps} fps, "
        f"{video.shape[2]}x{video.shape[3]}, audio {audio.shape[0]}/{sr}Hz)"
    )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ovi 1.1 — twin-backbone audio + video joint generation pipeline.

This pipeline integrates Character.AI's Ovi 1.1
(https://github.com/character-ai/Ovi, https://huggingface.co/chetwinlow1/Ovi)
into vLLM-Omni's diffusion engine. Ovi is a "Twin Backbone Cross-Modal
Fusion" model: a video DiT and an audio DiT run in lockstep with per-block
bidirectional cross-attention fusion, producing time-aligned video and
audio in a single denoising loop.

External components required at runtime (downloaded from Hugging Face the
first time the pipeline is constructed):

- chetwinlow1/Ovi                 — fusion DiT checkpoint (`od_config.model`)
- Wan-AI/Wan2.2-TI2V-5B           — UMT5 text encoder + WAN VAE 2.2 (video)
- hkchengrex/MMAudio              — MMAudio TOD VAE + BigVGAN vocoder

The selected fusion checkpoint variant is read from
`od_config.tf_model_config["variant"]` (default: "960x960_5s"). See
`NAME_TO_MODEL_SPECS_MAP` below for the supported variants.
"""

from __future__ import annotations

import os
import re
from typing import Any

import PIL.Image
import torch
import torch.nn as nn
from diffusers.video_processor import VideoProcessor
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportAudioInput, SupportImageInput
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .fusion import FusionModel

logger = init_logger(__name__)


# --- Model architecture configs (identical to Ovi/ovi/configs/model/dit/) ---

VIDEO_CONFIG: dict[str, Any] = {
    "patch_size": [1, 2, 2],
    "model_type": "ti2v",
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "num_heads": 24,
    "num_layers": 30,
    "in_dim": 48,
    "out_dim": 48,
    "text_len": 512,
    "window_size": [-1, -1],
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
}

AUDIO_CONFIG: dict[str, Any] = {
    "patch_size": [1],
    "model_type": "t2a",
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "num_heads": 24,
    "num_layers": 30,
    "in_dim": 20,
    "out_dim": 20,
    "text_len": 512,
    "window_size": [-1, -1],
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
    "temporal_rope_scaling_factor": 0.19676,
}


def _format_prompt_audcap(text: str) -> str:
    """720x720_5s model expects <AUDCAP>...<ENDAUDCAP> tags."""
    return re.sub(r"Audio:\s*(.*)", r"<AUDCAP>\1<ENDAUDCAP>", text, flags=re.S)


def _format_prompt_audio_colon(text: str) -> str:
    """960x960 models expect plain `Audio: ...` after a literal colon."""
    return re.sub(r"<AUDCAP>(.*?)<ENDAUDCAP>", r"Audio: \1", text, flags=re.S)


# Variant -> (fusion checkpoint filename, latent shapes, prompt formatter).
# Mirrors NAME_TO_MODEL_SPECS_MAP in Ovi/ovi/ovi_fusion_engine.py.
NAME_TO_MODEL_SPECS_MAP: dict[str, dict[str, Any]] = {
    "720x720_5s": {
        "path": "model.safetensors",
        "video_latent_length": 31,
        "audio_latent_length": 157,
        "video_area": 720 * 720,
        "formatter": _format_prompt_audcap,
    },
    "960x960_5s": {
        "path": "model_960x960.safetensors",
        "video_latent_length": 31,
        "audio_latent_length": 157,
        "video_area": 960 * 960,
        "formatter": _format_prompt_audio_colon,
    },
    "960x960_10s": {
        "path": "model_960x960_10s.safetensors",
        "video_latent_length": 61,
        "audio_latent_length": 314,
        "video_area": 960 * 960,
        "formatter": _format_prompt_audio_colon,
    },
}

# External HF repos that supply non-fusion components.
_WAN_REPO = "Wan-AI/Wan2.2-TI2V-5B"   # UMT5 + WAN VAE 2.2
_MMAUDIO_REPO = "hkchengrex/MMAudio"  # TOD VAE + BigVGAN vocoder
_MMAUDIO_TOD_VAE_PATH = "ext_weights/v1-16.pth"
_MMAUDIO_VOCODER_PATH = "ext_weights/best_netG.pt"


def _ensure_repo_local(repo_id: str, allow_patterns: list[str] | None = None) -> str:
    """Download a HF repo if not already cached. Returns the local snapshot path."""
    return snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns)


# --- pre / post-process functions registered via registry.py ---


def get_ovi_1_1_pre_process_func(od_config: OmniDiffusionConfig):
    """Pre-process: load and resize an optional input image for I2V mode."""

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            if isinstance(prompt, str):
                continue
            multi_modal_data = prompt.get("multi_modal_data") or {}
            raw_image = multi_modal_data.get("image")
            if raw_image is None:
                continue

            if isinstance(raw_image, str):
                image = PIL.Image.open(raw_image).convert("RGB")
            elif isinstance(raw_image, PIL.Image.Image):
                image = raw_image.convert("RGB")
            else:
                raise TypeError(
                    f"Unsupported image format {type(raw_image)}; expected file path or PIL.Image.Image."
                )

            # Default to 720P max area if dimensions weren't specified.
            sp = request.sampling_params
            if sp.height is None or sp.width is None:
                max_area = 720 * 1280
                aspect_ratio = image.height / image.width
                mod_value = 16
                h = round((max_area * aspect_ratio) ** 0.5) // mod_value * mod_value
                w = round((max_area / aspect_ratio) ** 0.5) // mod_value * mod_value
                if sp.height is None:
                    sp.height = h
                if sp.width is None:
                    sp.width = w

            image = image.resize((sp.width, sp.height), PIL.Image.Resampling.LANCZOS)
            prompt["multi_modal_data"]["image"] = image
            request.prompts[i] = prompt
        return request

    return pre_process_func


def get_ovi_1_1_post_process_func(od_config: OmniDiffusionConfig):
    """Post-process: pack video + audio outputs into a multi-modal payload."""

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(payload: Any, output_type: str = "np") -> dict[str, Any]:
        # `payload` is what pipeline.forward stuffed into DiffusionOutput.output.
        # We expect a dict with "video" and "audio" tensors (numpy arrays).
        if not isinstance(payload, dict):
            raise TypeError(
                f"Ovi 1.1 post-process expected a dict payload, got {type(payload).__name__}"
            )

        video = payload["video"]
        audio = payload["audio"]
        sample_rate = payload.get("audio_sample_rate", 16000)
        fps = payload.get("fps", 24)

        if output_type != "latent" and isinstance(video, torch.Tensor):
            video = video_processor.postprocess_video(video, output_type=output_type)

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().float().numpy()

        return {
            "video": video,
            "audio": audio,
            "audio_sample_rate": int(sample_rate),
            "fps": float(fps),
        }

    return post_process_func


# --- Composite scheduler dispatching to per-modality schedulers ---


class _VideoAudioScheduler:
    """Wraps the video and audio flow-matching schedulers so the diffuse loop
    can call a single ``.step((vid, aud), (t_v, t_a), (vid_lat, aud_lat))``
    instead of stepping each modality separately. Modeled after
    ``ltx2.pipeline_ltx2._VideoAudioScheduler``.
    """

    def __init__(self, video_scheduler, audio_scheduler):
        self.video_scheduler = video_scheduler
        self.audio_scheduler = audio_scheduler

    def step(self, noise_pred, t, latents, return_dict=False, generator=None):
        video_out = self.video_scheduler.step(
            noise_pred[0], t[0], latents[0], return_dict=False, generator=generator
        )[0]
        audio_out = self.audio_scheduler.step(
            noise_pred[1], t[1], latents[1], return_dict=False, generator=generator
        )[0]
        return ((video_out, audio_out),)


# --- Ovi 1.1 pipeline ---


class Ovi11Pipeline(nn.Module, CFGParallelMixin, SupportImageInput, SupportAudioInput):
    """Ovi 1.1 video + audio joint generation pipeline."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.target_dtype = od_config.dtype

        # --- Resolve variant + spec ---
        variant = od_config.tf_model_config.get("variant", "960x960_5s")
        if variant not in NAME_TO_MODEL_SPECS_MAP:
            raise ValueError(
                f"Unknown Ovi variant {variant!r}; expected one of "
                f"{list(NAME_TO_MODEL_SPECS_MAP)}"
            )
        spec = NAME_TO_MODEL_SPECS_MAP[variant]
        self.variant = variant
        self.video_latent_length: int = spec["video_latent_length"]
        self.audio_latent_length: int = spec["audio_latent_length"]
        self.target_area: int = spec["video_area"]
        self.text_formatter = spec["formatter"]
        self.audio_latent_channel: int = AUDIO_CONFIG["in_dim"]
        self.video_latent_channel: int = VIDEO_CONFIG["in_dim"]

        # --- External component repos: WAN (text + video VAE) and MMAudio ---
        # od_config.model is expected to be the Ovi fusion repo (chetwinlow1/Ovi).
        # We additionally pull WAN and MMAudio because they are not redistributed
        # in the Ovi repo. Override paths can be passed via tf_model_config:
        #   wan_model_path, mmaudio_model_path
        wan_model_path = od_config.tf_model_config.get(
            "wan_model_path", _ensure_repo_local(_WAN_REPO)
        )
        mmaudio_model_path = od_config.tf_model_config.get(
            "mmaudio_model_path",
            _ensure_repo_local(
                _MMAUDIO_REPO,
                allow_patterns=[_MMAUDIO_TOD_VAE_PATH, _MMAUDIO_VOCODER_PATH],
            ),
        )

        # --- Text encoder (UMT5, shared with Wan2.2 ti2v) ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            wan_model_path, subfolder="tokenizer"
        )
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            wan_model_path,
            subfolder="text_encoder",
            torch_dtype=self.target_dtype,
        ).to(self.device)
        self.tokenizer_max_length = AUDIO_CONFIG["text_len"]  # 512

        # --- Video VAE (WAN VAE 2.2) ---
        self.vae = DistributedAutoencoderKLWan.from_pretrained(
            wan_model_path,
            subfolder="vae",
            torch_dtype=self.target_dtype,
        ).to(self.device)
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 16

        # --- Audio VAE (MMAudio TOD + BigVGAN vocoder, vendored) ---
        from ._vendored.mmaudio_vae import load_mmaudio_vae

        self.audio_vae = load_mmaudio_vae(
            tod_vae_ckpt=os.path.join(mmaudio_model_path, _MMAUDIO_TOD_VAE_PATH),
            bigvgan_vocoder_ckpt=os.path.join(mmaudio_model_path, _MMAUDIO_VOCODER_PATH),
            mode="16k",
            need_vae_encoder=True,
        ).to(self.device)
        self.audio_vae.requires_grad_(False).eval()
        self.audio_sample_rate = 16000
        self.video_fps = 24

        # --- Fusion DiT (loaded via vllm-omni's standard weights pipeline) ---
        self.transformer = FusionModel(VIDEO_CONFIG, AUDIO_CONFIG)
        # Pick exactly the safetensors file matching the requested variant.
        # `prefix="transformer."` so AutoWeightsLoader maps the file's
        # top-level `video_model.*` / `audio_model.*` keys onto
        # `self.transformer.video_model.*` / `self.transformer.audio_model.*`.
        # Verified shape on H100 — see H100_HANDOFF D1.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,
                revision=getattr(od_config, "revision", None),
                prefix="transformer.",
                fall_back_to_pt=False,
                allow_patterns_overrides=[spec["path"]],
            )
        ]

        # --- CFG defaults (per modality) ---
        self.video_guidance_scale_default: float = 4.0
        self.audio_guidance_scale_default: float = 3.0
        self.slg_layer_default: int = 11
        self.flow_shift_default: float = 5.0
        self.num_inference_steps_default: int = 50

    # ------------------------------------------------------------------
    # Standard hooks
    # ------------------------------------------------------------------

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    @staticmethod
    def _make_scheduler(num_steps: int, shift: float, device) -> tuple[Any, torch.Tensor]:
        sched = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
        )
        sched.set_timesteps(num_steps, device=device, shift=shift)
        return sched, sched.timesteps

    def encode_prompt(
        self,
        prompts: list[str],
    ) -> list[torch.Tensor]:
        """Encode 3 prompts (positive, video-negative, audio-negative)."""
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.text_encoder(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
            )
        embeds = outputs.last_hidden_state
        return [embeds[i].to(self.target_dtype) for i in range(embeds.size(0))]

    def _encode_first_frame(
        self,
        image: PIL.Image.Image,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """VAE-encode the I2V first frame. Returns a [C, 1, H, W] latent."""
        from torchvision.transforms import functional as TVF

        tensor = TVF.to_tensor(image).to(self.device, dtype=self.target_dtype)
        tensor = (tensor - 0.5) * 2.0  # [-1, 1]
        tensor = tensor.unsqueeze(0).unsqueeze(2)  # [B=1, C, T=1, H, W]
        with torch.inference_mode():
            latents = self.vae.wrapped_encode(tensor) if hasattr(self.vae, "wrapped_encode") else self.vae.encode(tensor).latent_dist.mean
        return latents.to(self.target_dtype).squeeze(0)  # [C, 1, h, w]

    # ------------------------------------------------------------------
    # Denoising loop
    # ------------------------------------------------------------------

    def _diffuse(
        self,
        video_noise: torch.Tensor,
        audio_noise: torch.Tensor,
        timesteps_video: torch.Tensor,
        timesteps_audio: torch.Tensor,
        text_pos: torch.Tensor,
        text_video_neg: torch.Tensor,
        text_audio_neg: torch.Tensor,
        max_seq_len_video: int,
        max_seq_len_audio: int,
        scheduler_video,
        scheduler_audio,
        first_frame_clean: torch.Tensor | None,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        slg_layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The Ovi 1.1 denoising loop.

        Per step:
          1. (Optional I2V) re-inject the clean first-frame latent.
          2. Positive forward pass — single FusionModel call returns
             `(pred_vid_pos, pred_audio_pos)`.
          3. Negative forward pass — same call but with `slg_layer`
             skipping a transformer layer.
          4. Per-modality CFG combine with separate guidance scales.
          5. Step the video and audio schedulers independently.
        """
        is_i2v = first_frame_clean is not None
        for t_v, t_a in tqdm(zip(timesteps_video, timesteps_audio), total=len(timesteps_video)):
            timestep_input = torch.full((1,), t_v, device=self.device)

            if is_i2v:
                video_noise[:, :1] = first_frame_clean

            common_kwargs = {
                "vid": [video_noise],
                "audio": [audio_noise],
                "t": timestep_input,
                "vid_seq_len": max_seq_len_video,
                "audio_seq_len": max_seq_len_audio,
                "first_frame_is_clean": is_i2v,
            }
            pred_vid_pos, pred_audio_pos = self.transformer(
                vid_context=[text_pos],
                audio_context=[text_pos],
                **common_kwargs,
            )
            pred_vid_neg, pred_audio_neg = self.transformer(
                vid_context=[text_video_neg],
                audio_context=[text_audio_neg],
                slg_layer=slg_layer,
                **common_kwargs,
            )

            pred_vid_pos, pred_audio_pos = pred_vid_pos[0], pred_audio_pos[0]
            pred_vid_neg, pred_audio_neg = pred_vid_neg[0], pred_audio_neg[0]

            pred_video = pred_vid_neg + video_guidance_scale * (pred_vid_pos - pred_vid_neg)
            pred_audio = pred_audio_neg + audio_guidance_scale * (pred_audio_pos - pred_audio_neg)

            video_noise = scheduler_video.step(
                pred_video.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False
            )[0].squeeze(0)
            audio_noise = scheduler_audio.step(
                pred_audio.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False
            )[0].squeeze(0)

        if is_i2v:
            video_noise[:, :1] = first_frame_clean
        return video_noise, audio_noise

    # ------------------------------------------------------------------
    # Forward (entry point called by DiffusionEngine)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, request: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        if len(request.prompts) > 1:
            raise ValueError(
                "Ovi 1.1 supports only a single prompt per request; please pass "
                "one prompt object or string at a time."
            )
        r_prompt = request.prompts[0]

        if isinstance(r_prompt, str):
            prompt = r_prompt
            video_negative_prompt = ""
            audio_negative_prompt = ""
            multi_modal_data: dict[str, Any] = {}
        else:
            prompt = r_prompt.get("prompt", "")
            video_negative_prompt = r_prompt.get("video_negative_prompt", "")
            audio_negative_prompt = r_prompt.get("audio_negative_prompt", "")
            multi_modal_data = r_prompt.get("multi_modal_data") or {}

        if not prompt:
            raise ValueError("Ovi 1.1 requires a non-empty text prompt.")

        # Apply variant-specific audio prompt formatting.
        prompt = self.text_formatter(prompt)

        sp = request.sampling_params
        height = sp.height
        width = sp.width
        num_steps = sp.num_inference_steps or self.num_inference_steps_default
        shift = sp.extra_args.get("shift", self.flow_shift_default)
        seed = sp.seed if sp.seed is not None else 42

        video_guidance_scale = float(
            sp.extra_args.get("video_guidance_scale", self.video_guidance_scale_default)
        )
        audio_guidance_scale = float(
            sp.extra_args.get("audio_guidance_scale", self.audio_guidance_scale_default)
        )
        slg_layer = int(sp.extra_args.get("slg_layer", self.slg_layer_default))

        # If an input image is provided, switch to I2V mode.
        raw_image = multi_modal_data.get("image")
        is_i2v = raw_image is not None
        first_frame_clean: torch.Tensor | None = None

        # Snap dimensions to model-friendly multiples.
        if height is None or width is None:
            # Use the variant's target area with 1:1 aspect by default.
            side = int(self.target_area**0.5)
            height = side - side % 32
            width = height
        else:
            height = (height // 32) * 32
            width = (width // 32) * 32

        if is_i2v:
            assert isinstance(raw_image, PIL.Image.Image), (
                "Pre-process should have converted multi_modal_data['image'] to PIL.Image."
            )
            first_frame_clean = self._encode_first_frame(raw_image, height, width)
            # Match latent grid to the encoded first-frame.
            video_latent_h, video_latent_w = first_frame_clean.shape[-2], first_frame_clean.shape[-1]
        else:
            video_latent_h = height // self.vae_scale_factor_spatial
            video_latent_w = width // self.vae_scale_factor_spatial

        # Encode all three prompts in one batched call.
        prompt_embeds = self.encode_prompt(
            [prompt, video_negative_prompt, audio_negative_prompt]
        )
        text_pos, text_video_neg, text_audio_neg = prompt_embeds

        # Initial noise.
        gen = torch.Generator(device=self.device).manual_seed(seed)
        video_noise = torch.randn(
            (
                self.video_latent_channel,
                self.video_latent_length,
                video_latent_h,
                video_latent_w,
            ),
            device=self.device,
            dtype=self.target_dtype,
            generator=gen,
        )
        audio_noise = torch.randn(
            (self.audio_latent_length, self.audio_latent_channel),
            device=self.device,
            dtype=self.target_dtype,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )

        max_seq_len_audio = audio_noise.shape[0]
        ph, pw = (
            self.transformer.video_model.patch_size[1],
            self.transformer.video_model.patch_size[2],
        )
        max_seq_len_video = (
            video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (ph * pw)
        )

        scheduler_video, timesteps_video = self._make_scheduler(num_steps, shift, self.device)
        scheduler_audio, timesteps_audio = self._make_scheduler(num_steps, shift, self.device)

        with torch.amp.autocast(
            "cuda",
            enabled=self.target_dtype != torch.float32,
            dtype=self.target_dtype,
        ):
            video_noise, audio_noise = self._diffuse(
                video_noise=video_noise,
                audio_noise=audio_noise,
                timesteps_video=timesteps_video,
                timesteps_audio=timesteps_audio,
                text_pos=text_pos,
                text_video_neg=text_video_neg,
                text_audio_neg=text_audio_neg,
                max_seq_len_video=max_seq_len_video,
                max_seq_len_audio=max_seq_len_audio,
                scheduler_video=scheduler_video,
                scheduler_audio=scheduler_audio,
                first_frame_clean=first_frame_clean,
                video_guidance_scale=video_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                slg_layer=slg_layer,
            )

        # Decode video.
        video_latents_for_vae = video_noise.unsqueeze(0)
        decoded_video = (
            self.vae.wrapped_decode(video_latents_for_vae)
            if hasattr(self.vae, "wrapped_decode")
            else self.vae.decode(video_latents_for_vae).sample
        )
        decoded_video = decoded_video.squeeze(0).cpu().float().numpy()

        # Decode audio. Latents are [L, C]; MMAudio decode wants [B, C, L].
        audio_latents_for_vae = audio_noise.unsqueeze(0).transpose(1, 2)
        decoded_audio = self.audio_vae.wrapped_decode(audio_latents_for_vae)
        decoded_audio = decoded_audio.squeeze().cpu().float().numpy()

        return DiffusionOutput(
            output={
                "video": decoded_video,
                "audio": decoded_audio,
                "audio_sample_rate": self.audio_sample_rate,
                "fps": self.video_fps,
            }
        )

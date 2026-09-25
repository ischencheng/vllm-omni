# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Diffusion pipeline for Ming-Image Design and Design-Layer checkpoints."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import (
    DistributedAutoencoderKLQwenImage,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import (
    set_forward_context_direct_condition,
    set_forward_context_ref_latent,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch
from vllm_omni.diffusion.models.ming_image.condition import MingImageConditioning
from vllm_omni.diffusion.models.ming_image.request import (
    get_ming_image_pre_process_func as get_ming_image_pre_process_func,
)
from vllm_omni.diffusion.models.ming_image.request import (
    get_ming_image_prompt_extra,
    resolve_ming_image_request,
)
from vllm_omni.diffusion.models.ming_image.transformer import MingImageTransformer2DModel
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.hf_utils import get_diffusion_model_index
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = logging.getLogger(__name__)

_DESIGN_PIPELINE = "MingImageDiffusionPipeline"
_LAYERED_PIPELINE = "MingImageLayeredDiffusionPipeline"
_VENDOR_TRANSFORMER_CLASS = "DiffusionTransformer"
_DIFFUSION_REQUIRED_PATTERNS = [
    "model_index.json",
    "scheduler/**",
    "transformer/**",
    "vae/**",
    "mlp/**",
    "connector/**",
]


def _validate_variant_config(
    model_index: dict[str, Any] | None,
    transformer_config: Any,
) -> bool:
    """Validate checkpoint-owned variant metadata and return layered mode."""
    declared_transformer = getattr(transformer_config, "_class_name", None)
    if declared_transformer != _VENDOR_TRANSFORMER_CLASS:
        raise ValueError(
            "Ming-Image transformer/config.json must preserve the vendor "
            f"_class_name={_VENDOR_TRANSFORMER_CLASS!r}, got {declared_transformer!r}. "
            "vLLM-Omni loads those weights through MingImageTransformer2DModel."
        )

    alignment_padding_mode = getattr(transformer_config, "alignment_padding_mode", None)
    multi_frame_output = getattr(transformer_config, "multi_frame_output", None)
    variant = (alignment_padding_mode, multi_frame_output)
    if variant == ("zero_masked", False):
        is_layer_decomposition = False
    elif variant == ("learned", True):
        is_layer_decomposition = True
    else:
        raise ValueError(
            "Ming-Image transformer/config.json must use either "
            "alignment_padding_mode='zero_masked' with multi_frame_output=False "
            "or alignment_padding_mode='learned' with multi_frame_output=True; "
            f"got alignment_padding_mode={alignment_padding_mode!r}, "
            f"multi_frame_output={multi_frame_output!r}."
        )

    declared_pipeline = (model_index or {}).get("_class_name")
    if declared_pipeline is not None:
        if declared_pipeline not in {_DESIGN_PIPELINE, _LAYERED_PIPELINE}:
            raise ValueError(
                "Ming-Image model_index.json must declare "
                f"{_DESIGN_PIPELINE!r} or {_LAYERED_PIPELINE!r}, got {declared_pipeline!r}."
            )
        expected_pipeline = _LAYERED_PIPELINE if is_layer_decomposition else _DESIGN_PIPELINE
        if declared_pipeline != expected_pipeline:
            raise ValueError(
                f"Ming-Image model_index.json declares {declared_pipeline!r}, but "
                f"transformer/config.json describes {expected_pipeline!r}."
            )
    return is_layer_decomposition


class MingImageDiffusionPipeline(ZImagePipeline):
    """Ming-Image component adapter around the canonical Z-Image loop."""

    supports_request_batch = True

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["conditioning"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        del prefix
        nn.Module.__init__(self)

        model_path = od_config.model
        if not os.path.exists(model_path):
            model_path = download_weights_from_hf_specific(
                model_name_or_path=model_path,
                cache_dir=None,
                allow_patterns=_DIFFUSION_REQUIRED_PATTERNS,
                revision=od_config.revision,
                require_all=True,
            )
        local_files_only = os.path.isdir(model_path)
        dtype = od_config.dtype

        self.od_config = od_config
        self._execution_device = get_local_device()
        self.device = self._execution_device
        model_index = get_diffusion_model_index(model_path, revision=od_config.revision) or {}
        transformer_config = od_config.tf_model_config
        self.is_layer_decomposition = _validate_variant_config(
            model_index,
            transformer_config,
        )

        self._num_frames_per_prompt = 1
        self._pending_prompt_embeds: list[torch.Tensor] | None = None
        self._pending_negative_prompt_embeds: list[torch.Tensor] | None = None
        self._uses_cudagraph_trees = False

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]
        subfolders = ["scheduler", "transformer", "vae"]

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLQwenImage.from_pretrained,
            model_path,
            subfolder="vae",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)
        self.vae.eval()
        vae_channels = int(getattr(self.vae.config, "input_channels", 3))
        if vae_channels != 4:
            raise ValueError(f"Ming-Image requires an RGBA VAE, got input_channels={vae_channels}.")

        transformer_kwargs = get_transformer_config_kwargs(
            od_config.tf_model_config,
            MingImageTransformer2DModel,
        )
        self.transformer = MingImageTransformer2DModel(
            quant_config=od_config.quantization_config,
            **transformer_kwargs,
        )
        self.conditioning = MingImageConditioning(
            model_path,
            device=self.device,
            dtype=dtype,
        )
        self.text_encoder = None
        self.tokenizer = None

        self.vae_scale_factor = 2 ** len(self.vae.config.temperal_downsample)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2,
            do_convert_rgb=False,
        )
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler
        )

    def setup_compile(self) -> None:
        # Keep request preparation, scheduling, and VAE work eager,
        # while capturing repeated DiT blocks for CUDAGraph Trees replay.
        if self.od_config.diffusion_compile_granularity != "regional":
            logger.warning(
                "Ming-Image CUDA Graph uses regional DiT compilation; diffusion_compile_granularity=%r is ignored.",
                self.od_config.diffusion_compile_granularity,
            )
        self.transformer = regionally_compile(
            self.transformer,
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=self.od_config.diffusion_compile_dynamic,
        )
        self._uses_cudagraph_trees = True

    def encode_prompt(self, *args, **kwargs):
        del args, kwargs
        if self._pending_prompt_embeds is None or self._pending_negative_prompt_embeds is None:
            raise RuntimeError("Ming-Image conditioning is only available during forward.")
        return self._pending_prompt_embeds, self._pending_negative_prompt_embeds

    def prepare_latents(self, batch_size, *args, **kwargs):
        frames = self._num_frames_per_prompt
        flat = super().prepare_latents(batch_size * frames, *args, **kwargs)
        return torch.stack(flat.chunk(frames, dim=0), dim=2)

    def _encode_reference(
        self,
        reference: Any | None,
        height: int,
        width: int,
    ) -> torch.Tensor | None:
        if reference is None:
            return None
        if isinstance(reference, list):
            if len(reference) != 1:
                raise ValueError("Ming-Image currently accepts exactly one reference image.")
            reference = reference[0]
        if isinstance(reference, Image.Image):
            reference = reference.convert("RGBA")
        if not isinstance(reference, torch.Tensor):
            reference = self.image_processor.preprocess(reference, height=height, width=width)
        if reference.ndim != 4 or reference.shape[1] != 4:
            raise ValueError("Ming-Image reference must be a batched RGBA tensor or an RGBA-compatible image.")
        reference = reference.to(device=self.device, dtype=self.vae.dtype).unsqueeze(2)
        latent = self.vae.encode(reference).latent_dist.mode()
        return (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    def _configure_output_frames(
        self,
        *,
        reference: Any | None,
        num_layers: int,
        is_dummy_run: bool,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if self.is_layer_decomposition:
            if reference is None and not is_dummy_run:
                raise ValueError("Ming-Image Design-Layer requires a reference image.")
            self._num_frames_per_prompt = num_layers + 1
            return
        if num_layers != 1:
            raise ValueError("Ming-Image Design supports exactly one output frame.")
        self._num_frames_per_prompt = 1

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        if not req.requests:
            raise ValueError("Ming-Image requires at least one request.")
        for sampling in req.sampling_params_list:
            if sampling.num_outputs_per_prompt != 1:
                raise ValueError(
                    "Ming-Image currently supports num_outputs_per_prompt=1 only, "
                    f"got {sampling.num_outputs_per_prompt}."
                )
        if req.num_reqs > 1 and self.is_layer_decomposition:
            raise ValueError("Ming-Image Design-Layer does not support request batching.")

        settings = [
            resolve_ming_image_request(request, is_layer_decomposition=self.is_layer_decomposition)
            for request in req.requests
        ]
        first = settings[0]
        if any(item != first for item in settings[1:]):
            raise ValueError("Batched Ming-Image requests must use matching dimensions, steps, guidance and layers.")
        extras = [get_ming_image_prompt_extra(request.prompt) for request in req.requests]
        if req.num_reqs > 1 and any(extra.get("reference_image") is not None for extra in extras):
            raise ValueError("Ming-Image reference-image requests do not support request batching.")
        reference = extras[0].get("reference_image")
        self._configure_output_frames(
            reference=reference,
            num_layers=first.num_layers,
            is_dummy_run=req.is_dummy_run(),
        )

        positive = []
        direct_conditions = []
        inner_requests = []
        for request, extra in zip(req.requests, extras):
            sampling = request.sampling_params
            query_hidden = extra.get("query_hidden_states")
            direct_hidden = extra.get("direct_hidden_states")
            if query_hidden is None or direct_hidden is None:
                if not request.is_dummy_run():
                    raise ValueError("Ming-Image requests require query and direct conditions.")
                logger.warning("Ming-Image conditions are absent during warmup; using zero tensors.")
                query_hidden = torch.zeros((256, 2048), device=self.device, dtype=self.od_config.dtype)
                direct_hidden = torch.zeros((1, 6144), device=self.device, dtype=self.od_config.dtype)
            if not isinstance(query_hidden, torch.Tensor) or not isinstance(direct_hidden, torch.Tensor):
                raise TypeError("Ming-Image query and direct conditions must be tensors.")
            if query_hidden.ndim == 2:
                query_hidden = query_hidden.unsqueeze(0)
            if direct_hidden.ndim == 2:
                direct_hidden = direct_hidden.unsqueeze(0)
            if query_hidden.ndim != 3 or direct_hidden.ndim != 3:
                raise ValueError("Ming-Image conditions must have two or three dimensions.")
            if query_hidden.shape[0] != 1 or direct_hidden.shape[0] != 1:
                raise ValueError("Ming-Image conditions must contain exactly one request.")
            cap_feats, direct_condition = self.conditioning(
                query_hidden.to(device=self.device, dtype=self.od_config.dtype),
                direct_hidden.to(device=self.device, dtype=self.od_config.dtype),
            )
            positive.append(cap_feats[0])
            direct_conditions.append(direct_condition[0])

            seed = (sampling.extra_args or {}).get("seed", sampling.seed)
            generator = (
                torch.Generator(device=self.device).manual_seed(int(seed)) if seed is not None else sampling.generator
            )
            inner_requests.append(
                OmniDiffusionRequest(
                    prompt={"prompt": ""},
                    sampling_params=OmniDiffusionSamplingParams(
                        height=first.height,
                        width=first.width,
                        num_inference_steps=first.steps,
                        guidance_scale=first.cfg,
                        generator=generator,
                        output_type="latent",
                    ),
                    request_id=request.request_id,
                )
            )

        context_ref = self._encode_reference(reference, first.height, first.width)
        apply_cfg = first.cfg > 0
        context_direct = direct_conditions
        if apply_cfg:
            context_direct = direct_conditions + [torch.zeros_like(item) for item in direct_conditions]
            if context_ref is not None:
                context_ref = context_ref.repeat(2, 1, 1, 1, 1)

        self._pending_prompt_embeds = positive
        self._pending_negative_prompt_embeds = [torch.zeros_like(item) for item in positive]
        inner_req = DiffusionRequestBatch(requests=inner_requests)

        set_forward_context_ref_latent(context_ref)
        set_forward_context_direct_condition(context_direct)
        try:
            latent_output = super().forward(inner_req)
            if not isinstance(latent_output.output, torch.Tensor):
                raise TypeError("Ming-Image denoising must return latent tensors.")
            image = self._decode_latent_frames(latent_output.output)
            images = [image] if req.num_reqs == 1 else list(image.split(1, dim=0))
            return [DiffusionOutput(output=item, stage_durations=latent_output.stage_durations) for item in images]
        finally:
            set_forward_context_ref_latent(None)
            set_forward_context_direct_condition(None)
            self._pending_prompt_embeds = None
            self._pending_negative_prompt_embeds = None
            self._num_frames_per_prompt = 1

    @staticmethod
    def _flatten_latent_frames(latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 4:
            return latents
        if latents.ndim != 5:
            raise ValueError(f"Expected 4D or 5D Ming-Image latents, got {tuple(latents.shape)}")
        batch, channels, frames, height, width = latents.shape
        return latents.permute(2, 0, 1, 3, 4).reshape(
            frames * batch,
            channels,
            height,
            width,
        )

    def _decode_latent_frames(self, latents: torch.Tensor) -> torch.Tensor:
        latents = self._flatten_latent_frames(latents).to(self.vae.dtype).unsqueeze(2)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        if image.ndim == 5:
            if image.shape[2] != 1:
                raise ValueError(
                    f"Ming-Image VAE returned multiple decoded frames for a single latent frame: {tuple(image.shape)}"
                )
            image = image.squeeze(2)
        return image

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = AutoWeightsLoader(self).load_weights(weights)
        loaded |= {f"vae.{name}" for name, _ in self.vae.named_parameters()}
        loaded |= {f"conditioning.{name}" for name, _ in self.conditioning.named_parameters()}
        return loaded


def get_ming_image_post_process_func(od_config: OmniDiffusionConfig):
    del od_config
    image_processor = VaeImageProcessor(vae_scale_factor=16, do_convert_rgb=False)

    def post_process(images: torch.Tensor):
        if images.ndim == 5:
            images = images.permute(0, 2, 1, 3, 4).flatten(0, 1)
        return image_processor.postprocess(images.float())

    return post_process


__all__ = ["MingImageDiffusionPipeline", "get_ming_image_post_process_func"]

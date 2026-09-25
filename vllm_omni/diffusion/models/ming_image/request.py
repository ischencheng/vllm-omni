# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming-Image request settings and homogeneous batch admission."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniPromptType


@dataclass(frozen=True)
class MingImageRequestSettings:
    height: int
    width: int
    steps: int
    cfg: float
    num_layers: int


def get_ming_image_prompt_extra(prompt: OmniPromptType) -> dict[str, object]:
    if isinstance(prompt, dict):
        return dict(prompt.get("extra") or {})
    return {}


def get_ming_image_padded_condition_length(request: OmniDiffusionRequest) -> int | None:
    """Read the transformer caption length without materializing condition data."""
    extra = get_ming_image_prompt_extra(request.prompt)
    length = 0
    for key in ("query_hidden_states", "direct_hidden_states"):
        condition = extra.get(key)
        if not isinstance(condition, torch.Tensor) or condition.ndim not in (2, 3):
            return None
        if condition.ndim == 3 and condition.shape[0] != 1:
            return None
        if condition.shape[-2] == 0:
            return None
        length += condition.shape[-2]
    # Both condition projections preserve sequence length. The inherited
    # Z-Image transformer pads their concatenation to a multiple of 32.
    return length + (-length) % 32


def resolve_ming_image_request(
    request: OmniDiffusionRequest,
    *,
    is_layer_decomposition: bool,
) -> MingImageRequestSettings:
    """Resolve the same model-owned overrides before admission and forward."""
    sampling = request.sampling_params
    extra_args = sampling.extra_args or {}
    extra = get_ming_image_prompt_extra(request.prompt)
    default_cfg = 2.0 if is_layer_decomposition else 1.0
    return MingImageRequestSettings(
        height=int(extra_args.get("height") or sampling.height or 1024),
        width=int(extra_args.get("width") or sampling.width or 1024),
        steps=int(extra_args.get("steps") or sampling.num_inference_steps or 12),
        cfg=float(
            extra_args["cfg"]
            if extra_args.get("cfg") is not None
            else sampling.guidance_scale
            if sampling.guidance_scale is not None
            else default_cfg
        ),
        num_layers=int(extra_args.get("num_layers", extra.get("num_layers", 1))),
    )


def get_ming_image_pre_process_func(od_config: OmniDiffusionConfig):
    """Keep shared denoising controls homogeneous and unsupported modes single."""
    is_layer_decomposition = od_config.model_class_name == "MingImageLayeredDiffusionPipeline" or bool(
        getattr(od_config.tf_model_config, "multi_frame_output", False)
    )

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        settings = resolve_ming_image_request(request, is_layer_decomposition=is_layer_decomposition)
        reference = get_ming_image_prompt_extra(request.prompt).get("reference_image")
        condition_length = get_ming_image_padded_condition_length(request)
        # Missing conditions (including warmup), Design-Layer and reference
        # inputs retain their single-request path. The transformer requires
        # equal padded caption lengths to preserve each request's positions.
        singleton_id = (
            request.request_id if is_layer_decomposition or reference is not None or condition_length is None else None
        )
        request.batch_compatibility_key = (
            "ming_image",
            settings.height,
            settings.width,
            settings.steps,
            settings.cfg,
            settings.num_layers,
            condition_length,
            reference is not None,
            singleton_id,
        )
        return request

    return pre_process_func


__all__ = [
    "MingImageRequestSettings",
    "get_ming_image_padded_condition_length",
    "get_ming_image_pre_process_func",
    "get_ming_image_prompt_extra",
    "resolve_ming_image_request",
]

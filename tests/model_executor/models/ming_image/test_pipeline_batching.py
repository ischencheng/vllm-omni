# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass
from threading import Lock

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    override_forward_context,
)
from vllm_omni.diffusion.models.ming_image.pipeline import MingImageDiffusionPipeline
from vllm_omni.diffusion.models.ming_image.request import get_ming_image_padded_condition_length
from vllm_omni.diffusion.models.ming_image.transformer import MingImageTransformer2DModel
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Conditioning(torch.nn.Module):
    def forward(self, query, direct):
        # Retain request-specific values and lengths without loading weights.
        return query[..., :2], direct[..., :2]


@dataclass
class _VAEConfig:
    scaling_factor: float = 1.0
    shift_factor: float = 0.0


class _VAE(torch.nn.Module):
    dtype = torch.float32
    config = _VAEConfig()

    def __init__(self):
        super().__init__()
        self.decode_batch_sizes = []

    def decode(self, latents, return_dict):
        assert return_dict is False
        self.decode_batch_sizes.append(latents.shape[0])
        return (latents,)


class _Scheduler:
    config: dict[str, object] = {}
    order = 1

    def set_timesteps(self, num_inference_steps, device, **kwargs):
        self.timesteps = torch.linspace(900, 100, num_inference_steps, device=device)

    def step(self, noise, timestep, latents, return_dict):
        assert return_dict is False
        return (latents + noise * 0.1,)


@pytest.fixture
def pipeline(monkeypatch):
    pipeline = MingImageDiffusionPipeline.__new__(MingImageDiffusionPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = pipeline._execution_device = torch.device("cpu")
    pipeline.od_config = OmniDiffusionConfig(dtype=torch.float32)
    pipeline.is_layer_decomposition = False
    pipeline._num_frames_per_prompt = 1
    pipeline._pending_prompt_embeds = None
    pipeline._pending_negative_prompt_embeds = None
    pipeline._uses_cudagraph_trees = False
    pipeline.vae_scale_factor = 2
    pipeline.conditioning = _Conditioning()
    pipeline.vae = _VAE()
    pipeline.scheduler = _Scheduler()
    pipeline._profiler_lock = Lock()
    pipeline._stage_durations = {"denoise": 1.0}
    pipeline.transformer = MingImageTransformer2DModel.__new__(MingImageTransformer2DModel)
    torch.nn.Module.__init__(pipeline.transformer)
    pipeline.transformer.in_channels = 4
    calls = []

    def transformer_forward(self, x, timestep, cap_feats, *, ref_x, cap_feats_2, **kwargs):
        assert ref_x is None
        assert len(x) == len(timestep) == len(cap_feats) == len(cap_feats_2)
        calls.append((x, cap_feats, cap_feats_2))
        output = [
            latent * 0.125 + query.mean() + direct.mean() for latent, query, direct in zip(x, cap_feats, cap_feats_2)
        ]
        return output, None

    monkeypatch.setattr(ZImageTransformer2DModel, "forward", transformer_forward)
    return pipeline, calls


def _request(index, *, cfg=2.0, seed_source="sampling", direct_length=None):
    seed = 17 + index
    sampling = OmniDiffusionSamplingParams(
        height=8,
        width=8,
        num_inference_steps=2,
        guidance_scale=cfg,
        seed=seed,
    )
    if seed_source == "extra_args":
        sampling.seed = 999
        sampling.extra_args["seed"] = seed
    elif seed_source == "generator":
        sampling.seed = None
        sampling.generator = torch.Generator().manual_seed(seed)
    return OmniDiffusionRequest(
        prompt={
            "prompt": f"design {index}",
            "extra": {
                "query_hidden_states": torch.full((256, 2048), float(index + 1)),
                "direct_hidden_states": torch.full(
                    (2 + index * 3 if direct_length is None else direct_length, 6144), float(index + 11)
                ),
            },
        },
        sampling_params=sampling,
        request_id=f"design-{index}",
    )


@pytest.mark.parametrize("direct_length, padded_length", [(31, 288), (39, 320)])
def test_design_caption_buckets_match_real_transformer_geometry(direct_length, padded_length):
    """Admission must prevent per-row RoPE grids and outer padding from differing."""
    transformer = ZImageTransformer2DModel.__new__(ZImageTransformer2DModel)
    torch.nn.Module.__init__(transformer)
    (
        _,
        primary,
        _,
        image_positions,
        caption_positions,
        _,
        caption_padding,
        direct,
    ) = transformer.patchify_and_embed(
        [torch.zeros(4, 1, 8, 16), torch.zeros(4, 1, 8, 16)],
        [torch.ones(256, 2), torch.ones(256, 2)],
        patch_size=2,
        f_patch_size=1,
        all_cap_feats_2=[torch.ones(28, 2), torch.ones(direct_length, 2)],
    )

    captions = [torch.cat([query, prefix]) for query, prefix in zip(primary, direct)]
    assert [len(item) for item in captions] == [288, padded_length]
    requests = [_request(0, direct_length=28), _request(1, direct_length=direct_length)]
    assert [get_ming_image_padded_condition_length(request) for request in requests] == [len(item) for item in captions]
    assert [len(item) for item in caption_positions] == [288, padded_length]
    assert [item.sum().item() for item in caption_padding] == [4, padded_length - 256 - direct_length]
    assert image_positions[0][0, 0].item() == 289
    assert image_positions[1][0, 0].item() == padded_length + 1

    if padded_length == 288:
        # The shared rotary implementation may safely reuse the first row.
        torch.testing.assert_close(image_positions[0], image_positions[1], rtol=0, atol=0)
        torch.testing.assert_close(caption_positions[0], caption_positions[1], rtol=0, atol=0)
    else:
        assert not torch.equal(image_positions[0], image_positions[1])

    # Z-Image attention ignores its mask. Only equal padded lengths avoid
    # introducing additional keys that were absent in a singleton forward.
    batched_captions = pad_sequence(captions, batch_first=True)
    assert batched_captions.shape[1] - captions[0].shape[0] == padded_length - 288


def test_design_batch_accepts_different_direct_lengths_in_same_caption_bucket(pipeline):
    model, calls = pipeline
    requests = [_request(0, direct_length=28), _request(1, direct_length=31)]

    with override_forward_context(ForwardContext()):
        outputs = model.forward(DiffusionRequestBatch(requests=requests))

    assert len(outputs) == 2
    assert [item.shape[0] for item in calls[0][2]] == [28, 31, 28, 31]


@pytest.mark.parametrize("direct_lengths", [(28, 39), (39, 28)])
def test_design_batch_rejects_incompatible_caption_buckets_before_denoising(pipeline, direct_lengths):
    model, calls = pipeline
    requests = [_request(index, direct_length=length) for index, length in enumerate(direct_lengths)]
    context = ForwardContext()

    with override_forward_context(context), pytest.raises(ValueError, match="same padded condition length"):
        model.forward(DiffusionRequestBatch(requests=requests))

    assert not calls
    assert not model.vae.decode_batch_sizes
    assert context.direct_condition is None
    assert model._pending_prompt_embeds is None
    assert model._pending_negative_prompt_embeds is None


@pytest.mark.parametrize("cfg", [0.0, 2.0])
@pytest.mark.parametrize("seed_source", ["sampling", "extra_args", "generator"])
def test_design_batch_matches_singletons_with_independent_conditions_and_rng(pipeline, cfg, seed_source):
    """Exercise the real denoising loop, transformer adapter, and output splitting."""
    model, calls = pipeline
    requests = [_request(index, cfg=cfg, seed_source=seed_source) for index in range(2)]
    batch = DiffusionRequestBatch(requests=requests)
    context = ForwardContext()

    with override_forward_context(context):
        outputs = model.forward(batch)

    assert isinstance(outputs, list)
    assert len(outputs) == 2
    assert model.vae.decode_batch_sizes == [2]
    assert len(calls) == 2
    latents, queries, direct = calls[0]
    assert [item.shape[0] for item in direct] == ([2, 5, 2, 5] if cfg else [2, 5])
    assert [item.mean().item() for item in queries] == ([1, 2, 0, 0] if cfg else [1, 2])
    assert [item.mean().item() for item in direct] == ([11, 12, 0, 0] if cfg else [11, 12])
    for index in range(2):
        expected_noise = torch.randn((1, 4, 4, 4), generator=torch.Generator().manual_seed(17 + index))
        torch.testing.assert_close(latents[index], expected_noise[0].unsqueeze(1), rtol=0, atol=0)
        if cfg:
            torch.testing.assert_close(latents[index], latents[index + 2], rtol=0, atol=0)

    assert context.direct_condition is None
    assert context.ref_latent is None
    assert model._pending_prompt_embeds is None
    assert model._pending_negative_prompt_embeds is None
    assert model._num_frames_per_prompt == 1

    for index, output in enumerate(outputs):
        single = DiffusionRequestBatch(requests=[_request(index, cfg=cfg, seed_source=seed_source)])
        with override_forward_context(ForwardContext()):
            reference = model.forward(single)
        assert output.output.shape == (1, 4, 4, 4)
        assert output.stage_durations == model.stage_durations
        assert isinstance(reference, list)
        assert len(reference) == 1
        torch.testing.assert_close(output.output, reference[0].output, rtol=0, atol=0)


def test_layer_singleton_preserves_all_frames_in_one_request_output(pipeline):
    model, calls = pipeline
    model.is_layer_decomposition = True
    request = _request(0)
    request.request_id = DUMMY_DIFFUSION_REQUEST_ID
    request.sampling_params.extra_args["num_layers"] = 2

    with override_forward_context(ForwardContext()):
        outputs = model.forward(DiffusionRequestBatch(requests=[request]))

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert outputs[0].output.shape == (3, 4, 4, 4)
    assert outputs[0].stage_durations == model.stage_durations
    assert model.vae.decode_batch_sizes == [3]
    assert len(calls) == 2
    assert calls[0][0][0].shape == (4, 3, 4, 4)
    assert model._num_frames_per_prompt == 1


@pytest.mark.parametrize("num_outputs", [1, 2])
def test_z_image_singleton_preserves_scalar_generator(pipeline, monkeypatch, num_outputs):
    """Collating one scalar RNG into a list changes CUDA random draws."""
    model, calls = pipeline
    request = _request(0, cfg=0.0, seed_source="generator")
    request.sampling_params.num_outputs_per_prompt = num_outputs
    request.sampling_params.output_type = "latent"
    model._pending_prompt_embeds = [torch.ones(256, 2)]
    model._pending_negative_prompt_embeds = [torch.zeros(256, 2)]
    generators = []

    def prepare_latents(batch_size, channels, height, width, dtype, device, generator, latents):
        generators.append(generator)
        return ZImagePipeline.prepare_latents(
            model, batch_size, channels, height, width, dtype, device, generator, latents
        )

    monkeypatch.setattr(model, "prepare_latents", prepare_latents)
    context = ForwardContext(direct_condition=[torch.ones(2, 2) for _ in range(num_outputs)])
    with override_forward_context(context):
        output = ZImagePipeline.forward(model, DiffusionRequestBatch(requests=[request]))

    assert len(generators) == 1
    assert generators[0] is request.sampling_params.generator
    assert output.output.shape == (num_outputs, 4, 4, 4)
    expected_noise = torch.randn((num_outputs, 4, 4, 4), generator=torch.Generator().manual_seed(17))
    for index in range(num_outputs):
        torch.testing.assert_close(calls[0][0][index], expected_noise[index].unsqueeze(1), rtol=0, atol=0)


def test_design_batch_clears_conditions_after_denoising_error(pipeline, monkeypatch):
    model, _ = pipeline

    def fail(*args, **kwargs):
        raise RuntimeError("denoising failed")

    monkeypatch.setattr(ZImageTransformer2DModel, "forward", fail)
    context = ForwardContext()
    with override_forward_context(context), pytest.raises(RuntimeError, match="denoising failed"):
        model.forward(DiffusionRequestBatch(requests=[_request(0), _request(1)]))

    assert context.direct_condition is None
    assert context.ref_latent is None
    assert model._pending_prompt_embeds is None
    assert model._pending_negative_prompt_embeds is None
    assert model._num_frames_per_prompt == 1


def test_layer_pipeline_rejects_request_batch(pipeline):
    model, calls = pipeline
    model.is_layer_decomposition = True

    with pytest.raises(ValueError, match="Design-Layer"):
        model.forward(DiffusionRequestBatch(requests=[_request(0), _request(1)]))

    assert not calls


def test_design_pipeline_rejects_reference_image_batch(pipeline):
    model, calls = pipeline
    requests = [_request(0), _request(1)]
    requests[1].prompt["extra"]["reference_image"] = torch.zeros(1, 4, 8, 8)

    with pytest.raises(ValueError, match="reference"):
        model.forward(DiffusionRequestBatch(requests=requests))

    assert not calls


def test_design_batch_validates_output_count_for_every_request(pipeline):
    model, calls = pipeline
    requests = [_request(0), _request(1)]
    requests[1].sampling_params.num_outputs_per_prompt = 2

    with pytest.raises(ValueError, match="num_outputs_per_prompt=1 only, got 2"):
        model.forward(DiffusionRequestBatch(requests=requests))

    assert not calls

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.models.ming_image.request import (
    MingImageRequestSettings,
    get_ming_image_pre_process_func,
    resolve_ming_image_request,
)
from vllm_omni.diffusion.registry import get_diffusion_pre_process_func
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import DiffusionRequestStatus, RequestScheduler
from vllm_omni.diffusion.sched.request_scheduler import build_request_batch_sampling_params_key
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _config(*, layered=False, model_class_name="MingImageDiffusionPipeline"):
    return OmniDiffusionConfig(
        model="test",
        model_class_name=model_class_name,
        tf_model_config=TransformerConfig.from_dict({"multi_frame_output": layered}),
        max_num_seqs=2,
    )


def _request(request_id, *, seed=11, direct_length=4, extra_args=None, reference=None):
    extra = {
        "query_hidden_states": torch.full((256, 2048), float(seed)),
        "direct_hidden_states": torch.full((direct_length, 6144), float(seed)),
        "num_layers": 1,
    }
    if reference is not None:
        extra["reference_image"] = reference
    return OmniDiffusionRequest(
        prompt={"prompt": "", "extra": extra},
        sampling_params=OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=1.0,
            seed=seed,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            extra_args=extra_args or {},
        ),
        request_id=request_id,
    )


def _scheduler(config, *requests):
    pre_process = get_ming_image_pre_process_func(config)
    scheduler = RequestScheduler()
    scheduler.initialize(config)
    for request in requests:
        scheduler.add_request(pre_process(request))
    return scheduler


def test_request_settings_share_forward_override_precedence():
    request = _request(
        "a",
        extra_args={"height": "1024", "width": 768, "steps": "12", "cfg": 0.0, "num_layers": 3},
    )
    request.prompt["extra"]["num_layers"] = 2

    settings = resolve_ming_image_request(request, is_layer_decomposition=True)

    assert settings == MingImageRequestSettings(height=1024, width=768, steps=12, cfg=0.0, num_layers=3)


def test_request_settings_use_prompt_layer_count():
    request = _request("a")
    request.prompt["extra"]["num_layers"] = 4

    assert resolve_ming_image_request(request, is_layer_decomposition=True).num_layers == 4


@pytest.mark.parametrize(("layered", "expected_cfg"), [(False, 1.0), (True, 2.0)])
def test_request_settings_resolve_missing_values(layered, expected_cfg):
    request = _request("a")
    request.sampling_params.height = None
    request.sampling_params.width = None
    request.sampling_params.num_inference_steps = None
    request.sampling_params.guidance_scale = None

    settings = resolve_ming_image_request(request, is_layer_decomposition=layered)

    assert settings == MingImageRequestSettings(height=1024, width=1024, steps=12, cfg=expected_cfg, num_layers=1)


def test_different_seeds_and_ragged_direct_conditions_share_a_wave():
    first = _request("first", seed=11, direct_length=4, extra_args={"seed": 101})
    second = _request("second", seed=22, direct_length=7, extra_args={"seed": 202})
    scheduler = _scheduler(_config(), first, second)

    scheduled = scheduler.schedule()

    assert scheduled.scheduled_request_ids == ["first", "second"]
    assert build_request_batch_sampling_params_key(first) == build_request_batch_sampling_params_key(second)
    assert [request.req.sampling_params.extra_args["seed"] for request in scheduled.scheduled_new_reqs] == [101, 202]
    assert [request.req.sampling_params.generator.initial_seed() for request in scheduled.scheduled_new_reqs] == [
        11,
        22,
    ]


@pytest.mark.parametrize(
    "extra_args",
    [{"height": 1024}, {"width": 1024}, {"steps": 8}, {"cfg": 0.0}, {"num_layers": 2}],
)
def test_model_overrides_prevent_incompatible_requests_sharing_a_wave(extra_args):
    first = _request("first")
    second = _request("second", extra_args=extra_args)
    scheduler = _scheduler(_config(), first, second)

    scheduled = scheduler.schedule()

    assert scheduled.scheduled_request_ids == ["first"]
    assert scheduler.num_waiting_requests() == 1
    assert build_request_batch_sampling_params_key(first) != build_request_batch_sampling_params_key(second)


@pytest.mark.parametrize(
    "config",
    [_config(layered=True), _config(model_class_name="MingImageLayeredDiffusionPipeline")],
)
def test_layered_requests_retain_single_request_admission(config):
    scheduler = _scheduler(config, _request("first"), _request("second"))

    assert scheduler.schedule().scheduled_request_ids == ["first"]
    assert scheduler.num_waiting_requests() == 1


def test_reference_requests_retain_single_request_admission():
    reference = torch.zeros((1, 4, 16, 16))
    scheduler = _scheduler(
        _config(),
        _request("first", reference=reference),
        _request("second", reference=reference),
    )

    assert scheduler.schedule().scheduled_request_ids == ["first"]
    assert scheduler.num_waiting_requests() == 1


def test_reference_and_text_requests_do_not_share_a_wave():
    scheduler = _scheduler(
        _config(),
        _request("text"),
        _request("reference", reference=torch.zeros((1, 4, 16, 16))),
    )

    assert scheduler.schedule().scheduled_request_ids == ["text"]
    assert scheduler.num_waiting_requests() == 1


@pytest.mark.parametrize("request_id", ["first", "second"])
def test_aborted_request_does_not_change_its_batch_peer_result(request_id):
    scheduler = _scheduler(_config(), _request("first"), _request("second"))
    scheduled = scheduler.schedule()
    scheduler.finish_requests(request_id, DiffusionRequestStatus.FINISHED_ABORTED)
    outputs = BatchRunnerOutput.from_list(
        [
            RunnerOutput(
                request_id=rid,
                step_index=None,
                finished=True,
                result=DiffusionOutput(output=torch.tensor([index])),
            )
            for index, rid in enumerate(["first", "second"])
        ]
    )

    finished = scheduler.update_from_output(scheduled, outputs)

    assert finished == {"first", "second"}
    assert scheduler.get_request_state(request_id).status == DiffusionRequestStatus.FINISHED_ABORTED
    peer_id = "second" if request_id == "first" else "first"
    assert scheduler.get_request_state(peer_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
    assert outputs.get_request_output(peer_id).result.output.item() == (peer_id == "second")


@pytest.mark.parametrize("model_class_name", ["MingImageDiffusionPipeline", "MingImageLayeredDiffusionPipeline"])
def test_registry_loads_ming_image_admission_preprocessor(model_class_name):
    config = _config(model_class_name=model_class_name)
    request = _request("first")

    assert get_diffusion_pre_process_func(config)(request) is request
    assert request.batch_compatibility_key is not None

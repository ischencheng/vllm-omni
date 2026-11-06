# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniCPM-o deploy profiles whose CUDA Talker runs on MRv2 keep it single-frame."""

from pathlib import Path

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import (
    _apply_platform_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import _CODEC_EOS_TOKEN_ID, MINICPMO_4_5_PIPELINE
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY_DIR = Path(get_deploy_config_path("minicpmo_4_5.yaml")).parent
_PROFILES = sorted(path.name for path in _DEPLOY_DIR.glob("minicpmo_4_5*.yaml"))
_MRV2_TALKER_PROFILES = {
    "minicpmo_4_5_turn_mrv2.yaml",
    "minicpmo_4_5_turn_mrv2_h200.yaml",
    "minicpmo_4_5_duplex_mrv2.yaml",
}


def _cuda_talker(monkeypatch, name: str):
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(name)), platform="cuda")
    return next(stage for stage in merge_pipeline_deploy(MINICPMO_4_5_PIPELINE, config) if stage.stage_id == 1)


def test_mrv2_talker_profiles_exist() -> None:
    assert _MRV2_TALKER_PROFILES <= set(_PROFILES)


@pytest.mark.parametrize("name", _PROFILES)
def test_mrv2_talker_profiles_drop_multi_frame_decode(monkeypatch, name: str) -> None:
    talker = _cuda_talker(monkeypatch, name)
    on_mrv2 = bool(talker.yaml_engine_args.get("use_v2_model_runner"))
    assert on_mrv2 == (name in _MRV2_TALKER_PROFILES)
    if on_mrv2:
        # The base CUDA profile's multi-frame decode block is V1 only; the
        # MRv2 Talker refuses to load with it.
        assert talker.yaml_engine_args.get("speculative_config") is None
        # Its continue/stop id 1 is an ordinary codec id to a one-frame Talker.
        assert talker.yaml_extras["default_sampling_params"]["stop_token_ids"] == [_CODEC_EOS_TOKEN_ID]


@pytest.mark.parametrize("name", ["minicpmo_4_5_turn_mrv2.yaml", "minicpmo_4_5_turn_mrv2_h200.yaml"])
def test_turn_mrv2_talker_keeps_async_scheduling(monkeypatch, name: str) -> None:
    assert _cuda_talker(monkeypatch, name).yaml_engine_args["async_scheduling"] is True

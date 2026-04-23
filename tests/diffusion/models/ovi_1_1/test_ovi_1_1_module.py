# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structural tests for the Ovi 1.1 module.

These tests do NOT require GPU or model checkpoints — they only verify that
the module structure is correct (imports, registry wiring, config dicts).
End-to-end inference verification lives in the H100 handoff doc.
"""

import pytest

pytestmark = [pytest.mark.diffusion]


def test_ovi_1_1_imports():
    """Public symbols documented in __init__.py must be importable."""
    from vllm_omni.diffusion.models import ovi_1_1

    expected = {
        "AUDIO_CONFIG",
        "FusionModel",
        "NAME_TO_MODEL_SPECS_MAP",
        "Ovi11Pipeline",
        "VIDEO_CONFIG",
        "WanModel",
        "get_ovi_1_1_post_process_func",
        "get_ovi_1_1_pre_process_func",
    }
    missing = expected - set(ovi_1_1.__all__)
    assert not missing, f"Ovi 1.1 module missing exports: {missing}"


def test_audio_video_config_shapes():
    """Configs must match the reference (Ovi/ovi/configs/model/dit/)."""
    from vllm_omni.diffusion.models.ovi_1_1 import AUDIO_CONFIG, VIDEO_CONFIG

    assert VIDEO_CONFIG["dim"] == 3072
    assert VIDEO_CONFIG["num_layers"] == 30
    assert VIDEO_CONFIG["num_heads"] == 24
    assert VIDEO_CONFIG["in_dim"] == 48
    assert VIDEO_CONFIG["patch_size"] == [1, 2, 2]

    assert AUDIO_CONFIG["dim"] == 3072
    assert AUDIO_CONFIG["num_layers"] == 30
    assert AUDIO_CONFIG["num_heads"] == 24
    assert AUDIO_CONFIG["in_dim"] == 20
    assert AUDIO_CONFIG["patch_size"] == [1]

    # The two backbones must have matching block counts so per-block fusion
    # can pair them — this invariant is asserted in FusionModel.__init__.
    assert VIDEO_CONFIG["num_layers"] == AUDIO_CONFIG["num_layers"]


def test_three_variants_present():
    """All three Ovi variants must be in NAME_TO_MODEL_SPECS_MAP."""
    from vllm_omni.diffusion.models.ovi_1_1 import NAME_TO_MODEL_SPECS_MAP

    assert set(NAME_TO_MODEL_SPECS_MAP) == {"720x720_5s", "960x960_5s", "960x960_10s"}
    assert NAME_TO_MODEL_SPECS_MAP["960x960_10s"]["video_latent_length"] == 61
    assert NAME_TO_MODEL_SPECS_MAP["960x960_5s"]["video_latent_length"] == 31


def test_audio_prompt_formatters():
    """Audio prompt formatters must transform between the two tag styles."""
    from vllm_omni.diffusion.models.ovi_1_1 import NAME_TO_MODEL_SPECS_MAP

    # 720p model wants <AUDCAP> tags.
    fmt_720 = NAME_TO_MODEL_SPECS_MAP["720x720_5s"]["formatter"]
    assert fmt_720("A scene. Audio: birdsong") == "A scene. <AUDCAP>birdsong<ENDAUDCAP>"

    # 960p models want plain `Audio: ...`.
    fmt_960 = NAME_TO_MODEL_SPECS_MAP["960x960_10s"]["formatter"]
    assert fmt_960("A scene. <AUDCAP>birdsong<ENDAUDCAP>") == "A scene. Audio: birdsong"


def test_registry_entries_present():
    """Registry must list Ovi11Pipeline in the model + pre/post-process tables."""
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _DIFFUSION_PRE_PROCESS_FUNCS,
    )

    assert "Ovi11Pipeline" in _DIFFUSION_MODELS
    assert _DIFFUSION_MODELS["Ovi11Pipeline"] == ("ovi_1_1", "pipeline_ovi_1_1", "Ovi11Pipeline")
    assert _DIFFUSION_POST_PROCESS_FUNCS["Ovi11Pipeline"] == "get_ovi_1_1_post_process_func"
    assert _DIFFUSION_PRE_PROCESS_FUNCS["Ovi11Pipeline"] == "get_ovi_1_1_pre_process_func"


def test_mmaudio_vae_vendored_module_loads():
    """The vendored MMAudio VAE wrapper exposes the expected public API."""
    from vllm_omni.diffusion.models.ovi_1_1._vendored.mmaudio_vae import (
        MMAudioVAE,
        load_mmaudio_vae,
    )

    assert MMAudioVAE.__name__ == "FeaturesUtils"
    assert callable(load_mmaudio_vae)
    # Methods Ovi pipeline relies on
    assert hasattr(MMAudioVAE, "wrapped_encode")
    assert hasattr(MMAudioVAE, "wrapped_decode")

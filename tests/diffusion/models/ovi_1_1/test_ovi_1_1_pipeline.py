# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime/behavioral tests for the Ovi 1.1 pipeline.

Complements ``test_ovi_1_1_module.py`` (which covers imports, configs, and
registry wiring) with tests that exercise actual Python behavior:

* ``FusionModel`` structural integrity (block counts, cross-attn fusion
  injection) — the D7 runtime check, captured as a pytest so regressions
  surface in CI.
* ``rope_apply`` signature stability — guards against re-removal of the
  ``ref_lengths`` / ``freqs_scaling`` kwargs that inherited call sites
  rely on (see the [4/5] H100 fixes commit).
* ``_VideoAudioScheduler.step`` — verifies the composite scheduler
  dispatches each modality's pred/t/latent to the matching child.
* ``get_ovi_1_1_post_process_func`` — verifies the payload-to-mm-output
  transform packs every field the engine expects.

All tests are CPU-only and do not load any real checkpoint.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# D7 as a pytest: FusionModel instantiation + cross-attn fusion injection
# ---------------------------------------------------------------------------


def test_fusion_model_has_expected_block_count_and_fusion_modules() -> None:
    from vllm_omni.diffusion.models.ovi_1_1 import (
        AUDIO_CONFIG,
        VIDEO_CONFIG,
        FusionModel,
    )

    # Meta-device keeps the 11.66B parameter tensors zero-size so this
    # instantiation is fast and allocates nothing.
    with torch.device("meta"):
        model = FusionModel(VIDEO_CONFIG, AUDIO_CONFIG)

    assert len(model.video_model.blocks) == 30
    assert len(model.audio_model.blocks) == 30

    # Every video + audio cross-attn block must carry the 4 modules
    # injected by ``inject_cross_attention_kv_projections``. The Ovi
    # checkpoint ships with pre-trained weights for these tensors; if any
    # block lacks them, ``load_state_dict`` silently discards that half of
    # the fusion signal.
    required = ("k_fusion", "v_fusion", "pre_attn_norm_fusion", "norm_k_fusion")
    for i, block in enumerate(model.video_model.blocks):
        for name in required:
            assert hasattr(block.cross_attn, name), (
                f"video block {i} cross_attn missing injected module {name!r}"
            )
    for i, block in enumerate(model.audio_model.blocks):
        for name in required:
            assert hasattr(block.cross_attn, name), (
                f"audio block {i} cross_attn missing injected module {name!r}"
            )


# ---------------------------------------------------------------------------
# rope_apply signature regression guard (H100 fix [4/5] item 4)
# ---------------------------------------------------------------------------


def test_rope_apply_accepts_dreamid_style_kwargs() -> None:
    """``WanSelfAttention.forward`` (dreamid-derived) calls rope_apply with
    ``ref_lengths=None, freqs_scaling=None`` kwargs. The Ovi-derived
    function body only reads ``(x, grid_sizes, freqs)``, so both kwargs
    must be accepted-and-ignored. A TypeError here means someone
    removed the tolerant signature — re-adding it is the fix.
    """
    from vllm_omni.diffusion.models.ovi_1_1.wan_backbone import (
        rope_apply,
        rope_params,
    )

    x = torch.zeros(1, 4, 2, 16, dtype=torch.float32)  # [B, L, heads, d]
    grid_sizes = torch.tensor([[4]], dtype=torch.long)  # 1-D grid → rope_apply_1d
    # Use a small freq tensor covering fewer dims than head_dim // 2.
    freqs = rope_params(16, 4, freqs_scaling=0.5)

    # Positional call — baseline behavior.
    out = rope_apply(x, grid_sizes, freqs)
    assert out.shape == x.shape

    # Dreamid-style kwargs MUST be accepted.
    out_kw = rope_apply(x, grid_sizes, freqs, ref_lengths=None, freqs_scaling=None)
    assert out_kw.shape == x.shape


# ---------------------------------------------------------------------------
# Composite scheduler dispatches each modality to the matching child
# ---------------------------------------------------------------------------


class _StubChildScheduler:
    """Records the arguments it was called with so we can assert dispatch."""

    def __init__(self, return_tensor: torch.Tensor) -> None:
        self.calls: list[dict] = []
        self._return = return_tensor

    def step(self, noise_pred, t, latents, return_dict=False, generator=None):
        self.calls.append(
            {"noise_pred": noise_pred, "t": t, "latents": latents, "return_dict": return_dict}
        )
        # Mirror the diffusers SchedulerOutput tuple shape.
        return (self._return,)


def test_video_audio_scheduler_dispatches_correctly() -> None:
    from vllm_omni.diffusion.models.ovi_1_1.pipeline_ovi_1_1 import (
        _VideoAudioScheduler,
    )

    video_out = torch.full((1, 2), 7.0)
    audio_out = torch.full((1, 3), 11.0)

    video_child = _StubChildScheduler(video_out)
    audio_child = _StubChildScheduler(audio_out)
    composite = _VideoAudioScheduler(video_child, audio_child)

    pred_video = torch.full((1, 2), 1.0)
    pred_audio = torch.full((1, 3), 2.0)
    latent_video = torch.full((1, 2), 3.0)
    latent_audio = torch.full((1, 3), 4.0)
    t_video = torch.tensor(999)
    t_audio = torch.tensor(888)

    ((v_out, a_out),) = composite.step(
        (pred_video, pred_audio),
        (t_video, t_audio),
        (latent_video, latent_audio),
    )

    assert torch.equal(v_out, video_out)
    assert torch.equal(a_out, audio_out)

    # Each child should have been called exactly once with its own tuple
    # slot — no cross-wiring.
    assert len(video_child.calls) == 1
    assert torch.equal(video_child.calls[0]["noise_pred"], pred_video)
    assert torch.equal(video_child.calls[0]["latents"], latent_video)
    assert torch.equal(video_child.calls[0]["t"], t_video)

    assert len(audio_child.calls) == 1
    assert torch.equal(audio_child.calls[0]["noise_pred"], pred_audio)
    assert torch.equal(audio_child.calls[0]["latents"], latent_audio)
    assert torch.equal(audio_child.calls[0]["t"], t_audio)


# ---------------------------------------------------------------------------
# Post-process function packs the mm-output dict shape the engine consumes
# ---------------------------------------------------------------------------


def test_post_process_func_packs_expected_fields() -> None:
    from vllm_omni.diffusion.models.ovi_1_1 import get_ovi_1_1_post_process_func

    post_process = get_ovi_1_1_post_process_func(SimpleNamespace())

    # Simulate the pipeline's post-VAE output: video is a 5D [B, C, F, H, W]
    # tensor in [-1, 1]; audio is a 1-D numpy array. The post-process must
    # run VideoProcessor on the tensor (normalize + reshape) and pass the
    # audio through.
    video_tensor = torch.zeros(1, 3, 2, 4, 4)  # small fake decoded video
    audio = np.zeros(80384, dtype=np.float32)

    payload = {
        "video": video_tensor,
        "audio": audio,
        "audio_sample_rate": 16000,
        "fps": 24,
    }
    result = post_process(payload)

    assert set(result) == {"video", "audio", "audio_sample_rate", "fps"}
    assert result["audio_sample_rate"] == 16000
    # Default output_type is "np" — VideoProcessor.postprocess_video should
    # have converted the Tensor to a numpy [B, F, H, W, C] array in [0, 1].
    assert isinstance(result["video"], np.ndarray)
    assert result["video"].ndim == 5
    assert result["video"].shape[-1] == 3  # channels-last


def test_post_process_func_rejects_non_dict_payload() -> None:
    from vllm_omni.diffusion.models.ovi_1_1 import get_ovi_1_1_post_process_func

    post_process = get_ovi_1_1_post_process_func(SimpleNamespace())
    with pytest.raises(TypeError, match="expected a dict"):
        post_process(torch.zeros(1, 3, 2, 4, 4))

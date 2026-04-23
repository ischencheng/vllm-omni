# SPDX-License-Identifier: MIT
#
# Vendored from https://github.com/character-ai/Ovi (ovi/modules/mmaudio/),
# which itself ports MMAudio's audio VAE + BigVGAN vocoder. See LICENSES/
# at the parent directory for upstream attribution.

from pathlib import Path

from .features_utils import FeaturesUtils

MMAudioVAE = FeaturesUtils


def load_mmaudio_vae(
    tod_vae_ckpt: str | Path,
    bigvgan_vocoder_ckpt: str | Path,
    mode: str = "16k",
    need_vae_encoder: bool = True,
) -> MMAudioVAE:
    return MMAudioVAE(
        tod_vae_ckpt=str(tod_vae_ckpt),
        bigvgan_vocoder_ckpt=str(bigvgan_vocoder_ckpt),
        mode=mode,
        need_vae_encoder=need_vae_encoder,
    )


__all__ = ["MMAudioVAE", "load_mmaudio_vae"]

# Vendored upstream code attribution

This directory contains code vendored from third-party projects. Each subdirectory under `mmaudio_vae/` retains its original license file and full copyright notices alongside the vendored source.

## mmaudio_vae/

Source: https://github.com/character-ai/Ovi (`ovi/modules/mmaudio/`), which in turn ports the audio VAE + BigVGAN vocoder from MMAudio (https://github.com/hkchengrex/MMAudio).

License highlights:

- `mmaudio_vae/` itself — MIT (matches MMAudio upstream)
- `mmaudio_vae/ext/bigvgan/` — BigVGAN (NVIDIA), MIT. See `mmaudio_vae/ext/bigvgan/LICENSE` and `mmaudio_vae/ext/bigvgan/incl_licenses/LICENSE_{1..5}` for incorporated dependencies.
- `mmaudio_vae/ext/bigvgan_v2/` — BigVGAN v2 (NVIDIA), MIT. See `mmaudio_vae/ext/bigvgan_v2/LICENSE` and `mmaudio_vae/ext/bigvgan_v2/incl_licenses/LICENSE_{1..8}`.

Inference checkpoints `MMAudio/ext_weights/v1-16.pth` and `MMAudio/ext_weights/best_netG.pt` are downloaded at runtime from the `hkchengrex/MMAudio` Hugging Face repository and are not redistributed in this repository.

## Modifications from upstream

`mmaudio_vae/features_utils.py` has been minimally edited from the original Ovi `features_utils.py`:

- Removed unused `open_clip`, `create_model_from_pretrained`, and `torchvision.transforms.Normalize` imports.
- Removed the unused `patch_clip()` helper function.

These removals strip the CLIP-related path that is only relevant to MMAudio's text-to-audio training; the VAE encode/decode path used by Ovi 1.1 inference is unchanged. No model code or numerics are modified.

The CUDA-only `bigvgan_v2/alias_free_activation/cuda/` subdirectory is intentionally not vendored: BigVGAN v2 falls back to the PyTorch `alias_free_activation/torch/` path automatically when `use_cuda_kernel=False` (the default for inference).

The training-only files `ext/stft_converter.py`, `ext/stft_converter_mel.py`, `ext/rotary_embeddings.py`, `test_vae.py`, and `test_vae_config.yaml` are not vendored as they have zero references from the inference path.

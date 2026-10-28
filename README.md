# Raw results for vllm-project/vllm-omni#9319 (Moshi on the PersonaPlex duplex pipeline)

Everything here ran on the PR head `8bf25e88930ca7cf9a08a61d33c2a20a9f9a0d4a`, except `online-before-fix/`, which ran on the previous head `86efeb84fd1cd609aeac1f6ad4a80f2d90047ad3`.

- Image: `vllm/vllm-openai:v0.32.0` (vLLM 0.32.0, torch 2.13.0+cu130) plus `pip install .` of that revision.
- Checkpoints: `kyutai/moshiko-pytorch-bf16` at ``2bfc9ae6e89079a5cc7ed2a68436010d91a3d289``, `kyutai/moshika-pytorch-bf16` at ``a49141e28b3d9c947cf9aa5314431e1b11cbd2f5``.
- Reference: kyutai `moshi` 0.2.13 from PyPI, greedy, stepped the way `moshi.server` steps it (`scripts/ref_moshi.py`).
- Input: `hf-internal-testing/librispeech_asr_dummy` clean/validation `1272-128104-0000`, `-0001`, `-0002` (LibriSpeech, CC BY 4.0), resampled to 24 kHz, 0.5 s gaps, 6 s of silence at the end (`parity/input_24k.wav`).

| dir | what | hardware | script |
| --- | --- | --- | --- |
| `parity/` | two reference runs per checkpoint and one realtime session per arm, all in one container: `eager` (depformer graphs off), `graphs` (deploy default), `mrv2` (`personaplex_duplex_mrv2.yaml`), `repo_online` (served by repo id, Hub online), `repo_download_dir` (repo id with `--download-dir`, default Hub cache empty), `moshika` | 1x H100 80GB | `scripts/evidence.py` |
| `load/` | the driver's `--sessions 1/4/8/16` load mode, `nvidia-smi` memory samples | 1x H100 80GB, 24 CPUs | `scripts/load.py`, `scripts/summarize_load.py` |
| `e2e/` | `pytest -v --run-level=advanced_model tests/e2e/online_serving/test_personaplex_duplex.py`, Hub online | 1x H100 80GB | `scripts/e2e.py --online` |
| `unit/` | `pytest -m cpu` over the PersonaPlex tests | CPU | `scripts/unit.py` |
| `online-before-fix/` | `vllm serve kyutai/moshiko-pytorch-bf16` with the Hub online on `86efeb84f`: Stage 0 fails at load (401 on `nvidia/personaplex-7b-v1`) | 1x L40S | `scripts/repo_online.py` |

In `parity/`: `summary.json` has every comparison; `ref_<ckpt>_<a|b>/ref_server_greedy.{npz,json,wav}` are the reference tokens, text and audio; `vllm_<arm>.{npz,wav}` are this PR's per-step tokens and output audio; `trace_<arm>/*.jsonl` is the raw probe log (one line per Stage 0 step: text token and depformer codes per request); `server_<arm>.log` is the server log. Comparisons use `LMGen`'s output layout (step t's text and cb0 with step t+1's cb1..7), see `scripts/parity2.py:agreement` and `vllm_records`.

Volume paths in the logs are rewritten to `/cache`.

## Numbers in the PR, and where they come from

- `parity/summary.json`: `moshiko_ref_a_vs_ref_b` text 93.7% / whole frame 45.0%; every moshiko arm vs ref A 94.2% / 6.8%, vs ref B 91.6% / 6.8%; `moshika_ref_a_vs_ref_b` 91.1% / 39.5%; moshika arm vs its refs 91.1% / 93.5% text, 6.8% whole frame. The five moshiko arms produce the same Stage 0 tokens (`vllm_<arm>.npz`) and the same output audio (`vllm_<arm>.wav`), sample for sample.
- `parity/summary.json` `arms.repo_online` / `arms.repo_download_dir`: `stage0_runtimes` shows `moshi: true` with `model_path: kyutai/moshiko-pytorch-bf16`, and the server log has no request to `nvidia/personaplex-7b-v1`.
- `load/load-summary.json` (from `scripts/summarize_load.py load`): per-session frame latency, the driver's send lateness, pacing warning and continuity flags, and peak `nvidia-smi` memory.
- `e2e/e2e.log`: `1 passed, 1 skipped` (the PersonaPlex case needs `PERSONAPLEX_MODEL_PATH`).
- `unit/unit.log`: `339 passed, 1 skipped`.
- `online-before-fix/server_repo_online.log`: the 401 on `nvidia/personaplex-7b-v1/resolve/main/tokenizer-e351c8d8-checkpoint125.safetensors`.

## 2026-10-27 additions, on `3202166a7`

`3202166a7` makes Stage 0 decide Moshi from the weight names the talker loads; the commits after it only edit the recipe.

| dir | what | hardware | script |
| --- | --- | --- | --- |
| `numerics/` | kyutai's reference with only its numerics changed (bf16 twice, bf16 with cuBLAS reduced-precision reduction off, fp32) and one realtime session of this PR (deploy default), all in one container, same input as `parity/`. `ref_b2`/`ref_b4` (batch 2 and 4) failed in the script's decode step and have no results. | 1x H100 80GB | `scripts/sensitivity.py`, `scripts/ref_variant.py` |
| `unit-3202166a7/` | `pytest -m cpu` over the PersonaPlex tests: `340 passed, 1 skipped` | CPU | `scripts/unit.py` |
| `unit-3202166a7-old-talker/` | the same with `personaplex_talker.py` from `62d28ef0a`: `test_stage0_takes_moshi_from_the_weights_the_talker_loaded` fails | CPU | `scripts/unit.py` |
| `e2e-3202166a7/` | the online e2e: `1 passed, 1 skipped` | 1x H100 80GB | `scripts/e2e.py --online` |

Second reply in `numerics/` (first non-pad text step after step 40, from `sensitivity.json` and the `.npz` files):

| run | second reply starts at step | non-pad text tokens after step 40 |
| --- | --- | --- |
| reference bf16, run a | 134 | 19 |
| reference bf16, run b | 134 | 9 |
| reference bf16, reduced-precision reduction off | 134 | 13 |
| reference fp32 | 192 | 2 |
| this PR (bf16) | 192 | 2 |

Text agreement: this PR vs the fp32 reference 99.7% (first difference at step 192); bf16 run a vs run b 92.9%; fp32 vs bf16 run a 91.6%.

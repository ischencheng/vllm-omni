# MiniCPM-o turn handoff: Thinker rows by position (raw results)

Branch `fix/minicpmo-turn-talker-rows` (`a1877e70e` on `c7f56f015`). vLLM 0.32.0 (`vllm/vllm-openai:v0.32.0`), vllm-omni installed from the tree with `uv pip install .`, plus `step-audio2==1.0.0` for Token2Wav. Serving runs on 1x H100, unit tests on 1x L4.

Each serving run starts `vllm serve openbmb/MiniCPM-o-4_5 --omni --trust-remote-code --chat-template <minicpmo45_native.jinja> --chat-template-content-format openai --deploy-config <overlay>` (`overlay.yaml`, `server.log`, `results.json`, WAVs and Whisper transcripts per run). Command: `scripts/smoke.py --arms off --conc 1 --max-tokens 128 --audio [--base <profile>]`.

- `main-turn-mrv2-speech/`: upstream `8bcce4762` plus `handoff-dump.patch` (saves what `llm2tts` hands the Talker), `minicpmo_4_5_turn_mrv2.yaml`. 13/13 requests fail; Stage 1 logs `MiniCPM-o Talker condition length mismatch`. `dumps/` has the two handoffs built before Stage 1 stopped (latent rows vs tokens).
- `fix-turn-mrv2-speech/`: this branch, same profile. 13/13 return speech.
- `fix-v1-turn-speech/`: this branch, `minicpmo_4_5_turn.yaml` (V1 Thinker). 13/13.
- `unit/`: `scripts/unit.py` (default test set) on this branch.
- `unit-new-tests-on-main/`: this branch's `test_minicpmo_4_5_omni.py` against upstream `c7f56f015` code.

# MRv2 speculative decoding on the MiniCPM-o 4.5 Thinker (raw results)

Branch `feat/mrv2-propose-drafts` (head `636aecc44` on `c7f56f015`; its first commit is the turn-handoff fix on `fix/minicpmo-turn-talker-rows`). vLLM 0.32.0 (`vllm/vllm-openai:v0.32.0`), vllm-omni installed from the tree with `uv pip install .`, plus `step-audio2==1.0.0`. Serving runs on 1x H100, unit tests on 1x L4.

Each serving run starts `vllm serve openbmb/MiniCPM-o-4_5 --omni --trust-remote-code --chat-template <minicpmo45_native.jinja> --chat-template-content-format openai --deploy-config <overlay>`: `minicpmo_4_5_turn_mrv2.yaml` plus, for the DSpark arm, a stage-0 `speculative_config` (`dspark`, k=7, draft `Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm` at `c4c310aca2c768343165261cb22c21b90c1f79e0`). Each run directory has `overlay.yaml`, `server.log`, `results.json` and `metrics.txt`.

- `main-startup/`: the DSpark overlay on upstream `8bcce4762`; Stage 0 fails at load.
- `text-ab/`: `scripts/smoke.py --arms off,dspark --rounds 2 --conc 1,16 --max-tokens 256`, text only, on `f4d7c1beb` (same runner and model code as the head, before the stack moved onto the handoff fix; text-only requests don't reach `llm2tts`).
- `accepted-tokens-per-step/`: same prompts, `--arms dspark --conc 1,16 --max-tokens 256 --env SPEC_DEBUG=1` on `f4d7c1beb` plus `acceptance-print.patch`; `scripts/acceptance_phases.py dspark-r0/server.log` splits it by phase. Not a throughput run.
- `speech/`: `scripts/smoke.py --arms off,dspark --conc 1 --max-tokens 128 --audio --env SPEC_DUMP_DIR=<dir>` on the head plus `handoff-dump.patch`. `dumps/` holds each handoff (token ids + Thinker rows the Talker gets); `scripts/compare_handoff.py speech` writes `handoff_compare.json`. Text across containers can differ at near ties, so both arms run in one container.
- `unit/`: `scripts/unit.py` (default test set) on the head.
- `unit-new-tests-on-main/`: the head's runner and Thinker test files against upstream `c7f56f015` code.

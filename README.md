# MRv2 speculative decoding on the MiniCPM-o 4.5 Thinker: raw results

Branch `feat/mrv2-spec-decode` (head `f4d7c1beb`, base `8bcce4762`), vLLM 0.32.0 (`vllm/vllm-openai:v0.32.0`), vllm-omni installed from the tree with `uv pip install .`, plus `step-audio2==1.0.0` for Token2Wav. 1x H100 for the serving runs, 1x L4 for unit tests.

Every serving run starts `vllm serve openbmb/MiniCPM-o-4_5 --omni --trust-remote-code --chat-template <minicpmo45_native.jinja> --chat-template-content-format openai --deploy-config <overlay>`; each run directory has its `overlay.yaml`, `server.log`, `results.json` (per-request text, usage, latency) and `metrics.txt`. Draft: `Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm` at `c4c310aca2c768343165261cb22c21b90c1f79e0`, k=7.

- `text-ab/`: `scripts/smoke.py --arms off,dspark --rounds 2 --conc 1,16 --max-tokens 256` on the head. `summary.json` has tok/s per run and the greedy text comparison.
- `speech/`: `scripts/smoke.py --arms off,dspark --conc 1 --max-tokens 128 --audio` on the head. WAVs and Whisper (`openai/whisper-small.en`) transcripts per reply.
- `speech-without-accepted-row-slicing/`: same as `speech/` (DSpark arm) on the first commit only (`8aed77c07`).
- `accepted-tokens-per-step/`: `scripts/smoke.py --arms dspark --conc 1,16 --max-tokens 256 --env SPEC_DEBUG=1` on the head plus `acceptance-print.patch` (one `SPECACC` line per step with `num_sampled`). Not a throughput run.
- `main-startup/`: the DSpark overlay on upstream `8bcce4762`.
- `unit/`: `scripts/unit.py` (default test set) on the head.
- `unit-new-tests-on-main/`: the four touched test files from the head, run against upstream `8bcce4762` code.

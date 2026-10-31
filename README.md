# DSpark draft codes on the MiniCPM-o 4.5 MRv2 Talker (raw results)

Branch `feat/minicpmo-talker-dspark` (head `ffc7771e6`, two commits on #9696's `636aecc44`, base `c7f56f015`; serving runs on `fb442216e`, whose code is the same as the head's apart from one added test). vLLM 0.32.0 (`vllm/vllm-openai:v0.32.0`), vllm-omni installed from the tree with `uv pip install .`, plus `step-audio2==1.0.0`. Serving and unit runs on 1x H100.

Each serving run starts `vllm serve openbmb/MiniCPM-o-4_5 --omni --trust-remote-code --chat-template <minicpmo45_native.jinja> --chat-template-content-format openai --deploy-config <overlay>`: `minicpmo_4_5_turn_mrv2.yaml` plus, for the DSpark arm, a stage-1 `speculative_config` (`dspark`, k=7). No draft on the Thinker. Each run directory has `overlay.yaml`, `server.log`, `results.json` (per request: text, TTFA, latency, audio seconds, Whisper transcript and WER) and `metrics.txt`; audio files are left out.

Draft: `Qun1023/MiniCPM-o-4_5-talker-DSpark` at `0457ffd6980c2d1f19ebd3f1775f49764fa821af`, converted by `stage_draft()` in `scripts/talker_smoke.py`: the DSpark fields of `dflash_config` copied to the top level of `config.json`, and `embed_tokens.weight` (`tts.emb_code.0.weight`) and `lm_head.weight` (`tts.head_code.0`, weight norm restored, bf16) added from `openbmb/MiniCPM-o-4_5`.

- `timed-ab/container-{a,b}/`: `scripts/talker_smoke.py --arms off,dspark --rounds 2 --conc 1,16 --num-prompts 16 --c16-waves 4 --draft-sample probabilistic` (container b: `--arms dspark,off`), default Talker sampling with seed 42, on the head.
- `unseeded-c16/container-{a,b}/`: same with `--conc 16 --unseeded` (Talker `seed: null`).
- `argmax-draft-ab/`: `--arms off,dspark --rounds 2 --conc 1,16 --num-prompts 16 --c16-waves 2` (vLLM's default argmax drafts), on the head.
- `greedy-codes/`, `greedy-codes-6/`: `--greedy` (Talker `temperature: 0`) on the head plus `scripts/apply_debug.py`, which saves the codes handed to Code2Wav (`codes.jsonl`) and prints per-step `num_sampled` (`SPECACC`). `scripts/codes_by_prompt.py greedy-codes` compares codes per prompt across arms and phases.
- `codes-per-step/{argmax-draft,probabilistic-draft}/`: DSpark arm only, default sampling, debug tree; `scripts/analyze.py <dir>` splits `SPECACC` by phase. Not throughput runs.
- `unit/`: `scripts/unit.py` (default test set) on the head `ffc7771e6`.
- `new-tests-on-pr9696-code/`: the head's Talker and Thinker test files against #9696's code.

`scripts/analyze.py <run dir>` and `scripts/lengths.py <run dir> ...` produce the tables in the PR.

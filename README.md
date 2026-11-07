# MiniCPM-o turn MRv2: Thinker draft with the Thinker prefix cache

Raw results for vllm-project/vllm-omni#9864 (head `2b3ec91bd`, main `88cd44ad3`), 1x H100, vLLM 0.33.0.

- `ab-four-arms/`: four arms, two rounds in one container (`<arm>-r1` is the second round), plus `main-on-ds-r0`, which fails at startup. `analyze.txt` is `scripts/analyze.py` on it.
- `default-profile-and-one-token/`: `minicpmo_4_5_turn.yaml` with the draft on main and on the head, one-token speech replies, `/metrics` prefix cache counters in each `results.json`.
- `unit/`: CPU unit tests on the head (`head.log`) and the two changed test files on main (`main.log`).
- `duplex-e2e/`: the merge-lane MiniCPM-o duplex e2e on the head.
- `scripts/`: the driver, the analysis, and the debug edits applied to the measured trees (per-step acceptance print and Talker row dump, not part of the PR).

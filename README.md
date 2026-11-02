# MiniCPM-o turn handoff: ledger checks (raw results)

Branch `fix/minicpmo-turn-ledger-checks`, head `0c8198dc5` on `9990f10d9`, vLLM 0.32.0, 1x H100.

- `unit/`: `scripts/unit.py` (default test set): 2836 passed, 5 skipped.
- `serving/`: `scripts/ledger_check.py v1-prefix mrv2 dspark fault`, one server per case in one container. `results.json` has every request's status, text, transcript and WER, plus the counts of ledger failures, row warnings and length mismatches from the server log. `prefix-cache-metrics.txt` is the `/metrics` prefix cache counters.

`scripts/smoke.py` holds the prompts, request and scoring helpers `ledger_check.py` uses.

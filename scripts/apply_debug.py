"""Debug-only edits for a copy of the branch (never committed): SPECACC lines per
verify step (SPEC_DEBUG=1) and the Talker codes handed to Code2Wav (SPEC_DUMP_CODES).

python apply_debug.py <tree>
"""

import sys
from pathlib import Path

tree = Path(sys.argv[1])
p = tree / "vllm_omni/worker_v2/omni_ar_model_runner.py"
s = p.read_text()
old = "        if draft_inputs is not None:\n            self._propose_draft_tokens("
assert s.count(old) == 1
p.write_text(s.replace(old, (
    "        if draft_inputs is not None:\n"
    "            if __import__('os').environ.get('SPEC_DEBUG'):\n"
    "                print(f\"SPECACC nreq={input_batch.num_reqs} ntok={input_batch.num_tokens} \"\n"
    "                      f\"ndraft={input_batch.num_draft_tokens} \"\n"
    "                      f\"nsamp={num_sampled[: input_batch.num_reqs].tolist()}\", flush=True)\n"
    "            self._propose_draft_tokens("), 1))
p = tree / "vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py"
s = p.read_text()
old = "    new_codes = pending[:new_token_count]\n"
assert s.count(old) == 1
p.write_text(s.replace(old, old + (
    "    if __import__('os').environ.get('SPEC_DUMP_CODES'):\n"
    "        with open(__import__('os').environ['SPEC_DUMP_CODES'], 'a') as _f:\n"
    "            _f.write(__import__('json').dumps({'req': request_id, 'codes': [int(c) for c in new_codes],\n"
    "                                               'finished': bool(flush_pending)}) + '\\n')\n"), 1))
print("debug edits applied to", tree)

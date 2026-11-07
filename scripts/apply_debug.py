"""Debug-only edits for a scratch copy of a tree (never committed).

* SPECACC lines per verify step (SPEC_DEBUG=1): batch size, scheduled tokens,
  draft tokens and num_sampled per request, on the MRv2 AR runner.
* The rows llm2tts hands the Talker (SPEC_DUMP_ROWS=<dir>), one .pt per request.

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
anchor = """                tts_token_ids_slice = torch.tensor(
                    full_token_ids[tts_bos_idx : tts_bos_idx + tts_hidden_slice.shape[0]], dtype=torch.long
                )
"""
assert s.count(anchor) == 1
hook = anchor + """                import hashlib as _h, os as _o, time as _t
                _d = _o.environ.get("SPEC_DUMP_ROWS")
                if _d:
                    torch.save({"prompt_hash": _h.sha1(str(prompt_token_ids).encode()).hexdigest(),
                                "prompt_len": len(prompt_token_ids), "ids": full_token_ids[tts_bos_idx:end_idx],
                                "rows": tts_hidden_slice.float().cpu(), "time": _t.time()},
                               f"{_d}/{llm_output.request_id}.pt")
"""
p.write_text(s.replace(anchor, hook, 1))
print("debug edits applied to", tree)

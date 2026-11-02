"""Turn handoff checks for the malformed-ledger failure, one server per case, one container.

python ledger_check.py [v1-prefix] [mrv2] [dspark] [fault]  ->  $OUT_DIR/<case>-r0/results.json, server.log

v1-prefix: minicpmo_4_5_turn.yaml as shipped (stage 0 prefix cache on), every prompt twice, so the
           second pass hits the cache. The ledger check must not fire on cached rows.
mrv2:      minicpmo_4_5_turn_mrv2.yaml, every prompt once, replies cut at max_tokens included.
dspark:    minicpmo_4_5_turn_mrv2.yaml with a stage-0 DSpark draft, every prompt once.
fault:     minicpmo_4_5_turn_mrv2.yaml with the installed module patched so every third handoff gets a
           short ledger. Those requests must fail alone and the server must keep serving.
"""

import base64
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke  # noqa: E402

OUT = smoke.OUT
PORT = smoke.PORT
ANCHOR = '    row_positions = mm_output.get("latent_positions")\n    num_rows = int(thinker_hidden_states.shape[0])\n'
FAULT = (
    '    global _spec_fault_calls\n'
    '    _spec_fault_calls = globals().get("_spec_fault_calls", 0) + 1\n'
    '    if _spec_fault_calls % 3 == 0:\n'
    '        row_positions = row_positions[:-1]\n'
)
CASES = {
    "v1-prefix": {"base": "minicpmo_4_5_turn.yaml", "passes": 2, "prompts": None},
    "mrv2": {"base": "minicpmo_4_5_turn_mrv2.yaml", "passes": 1, "prompts": None},
    "dspark": {"base": "minicpmo_4_5_turn_mrv2.yaml", "passes": 1, "prompts": None},
    "fault": {"base": "minicpmo_4_5_turn_mrv2.yaml", "passes": 1, "prompts": 6},
}


def patch_fault():
    import vllm_omni

    path = Path(vllm_omni.__file__).resolve().parent / "model_executor/stage_input_processors/minicpmo_4_5_omni.py"
    text = path.read_text()
    assert text.count(ANCHOR) == 1, "turn handoff anchor not found"
    path.write_text(text.replace(ANCHOR, ANCHOR + FAULT))


def run_case(case):
    cfg = CASES[case]
    smoke.NUM_PROMPTS = cfg["prompts"]
    out = OUT / f"{case}-r0"
    out.mkdir(parents=True, exist_ok=True)
    if case == "fault":
        patch_fault()
    deploy = smoke.write_overlay("dspark", 7) if case == "dspark" else smoke.deploy_dir() / cfg["base"]
    template = smoke.deploy_dir().parent / "transformers_utils" / "chat_templates" / "minicpmo45_native.jinja"
    cmd = [sys.executable, "-m", "vllm_omni.entrypoints.cli.main", "serve", smoke.MODEL, "--omni",
           "--host", "127.0.0.1", "--port", str(PORT), "--deploy-config", str(deploy),
           "--trust-remote-code", "--chat-template", str(template), "--chat-template-content-format", "openai"]
    log = open(out / "server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result = {"arm": case, "round": 0, "base": cfg["base"]}
    try:
        ok, startup = smoke.wait_ready(proc, 2400)
        result["ready"], result["startup_s"] = ok, startup
        print(f"[{case}] ready={ok} after {startup:.0f}s", flush=True)
        if not ok:
            return result
        for n in range(1, cfg["passes"] + 1):
            rows, wall, toks = smoke.run_batch(smoke.prompt_set(), 128, 1)
            for i, r in enumerate(rows):
                data = r.pop("audio_b64", None)
                if data:
                    (out / f"c{n}_{i:02d}.wav").write_bytes(base64.b64decode(data))
                    r["audio_file"] = f"c{n}_{i:02d}.wav"
            result[f"c{n}"] = {"wall_s": wall, "completion_tokens": toks, "rows": rows}
            print(f"[{case}] pass {n}: status {[r['status'] for r in rows]} audio {sum(1 for r in rows if r.get('audio_file'))}"
                  f" cached_tokens {[((r.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens') for r in rows]}",
                  flush=True)
        if case == "fault":
            result["health_after"] = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=10).status_code
            after = smoke.one_request(smoke.prompt_set()[0], 32)
            result["request_after"] = {"status": after["status"], "has_audio": bool(after.get("audio_b64"))}
            print(f"[{case}] health {result['health_after']} next request {result['request_after']}", flush=True)
            for r in result["c1"]["rows"]:
                if r["status"] != 200:
                    print(f"[{case}] {r['status']} {r.get('error', '')[:300]}", flush=True)
        try:
            (out / "metrics.txt").write_text(requests.get(f"http://127.0.0.1:{PORT}/metrics", timeout=10).text)
        except requests.RequestException as e:
            (out / "metrics.txt").write_text(f"metrics unavailable: {e}\n")
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        log.close()
        time.sleep(5)
        text = (out / "server.log").read_text(errors="replace")
        result["malformed_lines"] = len(re.findall(r"malformed Thinker row ledger", text))
        result["rows_stop_lines"] = [ln[:220] for ln in text.splitlines() if "Thinker rows stop at token" in ln]
        result["length_mismatch_lines"] = len(re.findall(r"condition length mismatch", text))
        (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"[{case}] malformed={result['malformed_lines']} rows_stop={len(result['rows_stop_lines'])} "
              f"length_mismatch={result['length_mismatch_lines']}", flush=True)
    return result


def main():
    smoke.AUDIO = True
    for case in sys.argv[1:] or list(CASES):
        run_case(case)
    scores = smoke.score_audio()
    (OUT / "summary.json").write_text(json.dumps({"audio_wer": scores}, indent=2))


if __name__ == "__main__":
    main()

"""GPU e2e: pytest the JoyAI realtime duplex e2e tests (default profiles) in the image's vllm-omni tree.

python e2e.py  ->  $OUT_DIR/e2e.xml, e2e.log
"""

import os
import subprocess
import sys
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
SRC = Path(os.environ.get("VLLM_OMNI_TREE", "/opt/vllm-omni"))
TESTS = ["tests/e2e/online_serving/test_joyai_vl_interaction.py", "tests/e2e/online_serving/test_joyai_vl_delegation.py"]


def main():
    # --online: leave HF_HUB_OFFLINE unset, so the server keeps the repo id (as in CI).
    online = "--online" in sys.argv[1:]
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"} if online else {**os.environ, "HF_HUB_OFFLINE": "1"}
    cmd = [sys.executable, "-m", "pytest", "-v", "-s", "-p", "no:cacheprovider", "--run-level=advanced_model", f"--junitxml={OUT / 'e2e.xml'}", *TESTS]
    print("+", " ".join(cmd), flush=True)
    with open(OUT / "e2e.log", "w") as log:
        r = subprocess.run(cmd, cwd=SRC, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    lines = (OUT / "e2e.log").read_text().splitlines()
    print("\n".join(line for line in lines if any(k in line for k in ("PASSED", "FAILED", "SKIPPED", "ERROR", "passed", "failed")))[-3000:])
    print(f"rc={r.returncode}", flush=True)


if __name__ == "__main__":
    main()

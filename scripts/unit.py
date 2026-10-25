"""CPU unit tests of the PersonaPlex / Moshi duplex code in the image's vllm-omni tree.

python unit.py  ->  $OUT_DIR/unit.xml, unit.log
"""

import os
import subprocess
import sys
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
SRC = Path("/opt/vllm-omni")
TESTS = [
    "tests/model_executor/models/personaplex",
    "tests/model_executor/stage_input_processors/test_personaplex.py",
    "tests/engine/duplex/test_session_runner_personaplex.py",
    "tests/config/test_personaplex_duplex_mrv2_deploy.py",
]


def main():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    cmd = [sys.executable, "-m", "pytest", "-q", "-m", "cpu", "-p", "no:cacheprovider",
           f"--junitxml={OUT / 'unit.xml'}", *[t for t in TESTS if (SRC / t).exists()]]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=SRC, env=env, text=True, capture_output=True)
    (OUT / "unit.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    print("\n".join(r.stdout.strip().splitlines()[-30:]), flush=True)
    print(f"rc={r.returncode}", flush=True)


if __name__ == "__main__":
    main()

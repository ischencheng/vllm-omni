"""CPU unit tests in the image's vllm-omni tree.

python unit.py [test paths ...]  ->  $OUT_DIR/unit.xml, unit.log
"""

import os
import subprocess
import sys
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
SRC = Path(os.environ.get("VLLM_OMNI_TREE", "/opt/vllm-omni"))
DEFAULT = [
    "tests/worker_v2",
    "tests/model_executor/models/minicpmo_4_5",
    "tests/model_executor/models/moss_tts/test_local_eager_mtp.py",
    "tests/model_executor/stage_input_processors",
]


def main():
    tests = sys.argv[1:] or DEFAULT
    env = dict(os.environ)
    cmd = [sys.executable, "-m", "pytest", "-q", "-m", "cpu", "-p", "no:cacheprovider",
           f"--junitxml={OUT / 'unit.xml'}", *[t for t in tests if (SRC / t.split("::")[0]).exists()]]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=SRC, env=env, text=True, capture_output=True)
    (OUT / "unit.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    print("\n".join(r.stdout.strip().splitlines()[-40:]), flush=True)
    print(f"rc={r.returncode}", flush=True)


if __name__ == "__main__":
    main()

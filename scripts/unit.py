"""CPU unit tests of the JoyAI duplex and Qwen3-TTS stage code in the image's vllm-omni tree.

python unit.py  ->  $OUT_DIR/unit.xml, unit.log
"""

import os
import subprocess
import sys
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
SRC = Path(os.environ.get("VLLM_OMNI_TREE", "/opt/vllm-omni"))
TESTS = [
    "tests/model_executor/models/joyai_vl_interaction",
    "tests/model_executor/stage_input_processors/test_joyai_vl_interaction.py",
    "tests/config/test_config_merge.py",
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

"""CPU unit tests: the branch tree (installed at /opt/vllm-omni), then its changed
test files run against main (/opt/trees/main) to show which cases fail there.

python unit_ab.py  ->  $OUT_DIR/{head,main}.log, {head,main}.xml
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
HEAD = Path("/opt/vllm-omni")
MAIN = Path("/opt/trees/main")
CHANGED = [
    "tests/model_executor/models/minicpmo_4_5/test_thinker_mrv2.py",
    "tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py",
]
HEAD_TESTS = CHANGED + [
    "tests/model_executor/models/minicpmo_4_5",
    "tests/model_executor/stage_input_processors",
    "tests/worker_v2/test_prefix_cache.py",
    "tests/core/test_prefix_cache.py",
    "tests/config/test_config_factory.py",
]


def run(tag, tree, tests):
    env = dict(os.environ, PYTHONPATH=str(tree))
    cmd = [sys.executable, "-m", "pytest", "-q", "-m", "cpu", "-p", "no:cacheprovider",
           f"--junitxml={OUT / f'{tag}.xml'}", *[t for t in dict.fromkeys(tests) if (tree / t).exists()]]
    print("+", tag, " ".join(cmd[2:]), flush=True)
    r = subprocess.run(cmd, cwd=tree, env=env, text=True, capture_output=True)
    (OUT / f"{tag}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    print("\n".join(r.stdout.strip().splitlines()[-25:]), flush=True)
    print(f"{tag} rc={r.returncode}", flush=True)


def main():
    run("head", HEAD, HEAD_TESTS)
    for t in CHANGED:
        shutil.copy(HEAD / t, MAIN / t)
    run("main", MAIN, CHANGED)


if __name__ == "__main__":
    main()

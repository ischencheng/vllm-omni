"""GPU e2e: the MiniCPM-o duplex lanes from .buildkite/cuda in the image's vllm-omni tree.

python duplex_e2e.py [merge] [mrv2]  ->  $OUT_DIR/e2e-<lane>.xml, e2e-<lane>.log

merge = test-merge.yml "Omni · MiniCPM-o 4.5 Duplex Test" (runs only after merge).
mrv2  = test-ready.yml "Omni · MiniCPM-o 4.5 MRv2 Duplex Test".
"""

import os
import subprocess
import sys
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR", "/work/out"))
SRC = Path("/opt/vllm-omni")
LANES = {
    "merge": [
        "tests/e2e/online_serving/test_minicpmo_4_5_duplex.py",
        "tests/e2e/online_serving/test_minicpmo_4_5_window.py",
    ],
    "mrv2": ["tests/e2e/online_serving/test_minicpmo_4_5_duplex_mrv2.py"],
}
KEYS = ("PASSED", "FAILED", "SKIPPED", "ERROR", " passed", " failed", " error")


def main():
    for lane in sys.argv[1:] or ["merge"]:
        junit = OUT / f"e2e-{lane}.xml"
        log_path = OUT / f"e2e-{lane}.log"
        cmd = [
            sys.executable, "-m", "pytest", "-s", "-v", "-p", "no:cacheprovider",
            "-m", "advanced_model and cuda", "--run-level", "advanced_model",
            f"--junitxml={junit}", *LANES[lane],
        ]
        print("+", " ".join(cmd), flush=True)
        with open(log_path, "w") as log:
            r = subprocess.run(cmd, cwd=SRC, env=dict(os.environ), text=True, stdout=log, stderr=subprocess.STDOUT)
        lines = log_path.read_text(errors="replace").splitlines()
        print("\n".join(line[:200] for line in lines if any(k in line for k in KEYS))[-4000:], flush=True)
        print(f"lane={lane} rc={r.returncode}", flush=True)


if __name__ == "__main__":
    main()

"""Tokens committed per verify step, split by benchmark phase, from SPECACC lines.

python acceptance_phases.py <server.log>

One SPECACC line per runner step (acceptance-print.patch): batch size, scheduled
tokens, draft tokens and num_sampled per request. Phases follow the smoke.py
order: server startup warm-up (7 lines), the 4-prompt warm-up (11 lines), the
16 sequential c1 requests (783 lines), then the c16 batch to the end. Full
verify steps are those with 7 drafts and 8 rows for every request.
"""

import re
import sys

rows = []
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"SPECACC nreq=(\d+) ntok=(\d+) ndraft=(\d+) nsamp=\[([^\]]*)\]", line)
    if m:
        nreq, ntok, ndraft = map(int, m.groups()[:3])
        rows.append((nreq, ntok, ndraft, [int(x) for x in m.group(4).split(",") if x.strip()]))
phases = {"startup-warmup": rows[0:7], "warmup": rows[7:18], "c1": rows[18:801], "c16": rows[801:]}
for name, phase in phases.items():
    steps = tokens = 0
    for nreq, ntok, ndraft, sampled in phase:
        if ndraft == 7 * nreq and ntok == 8 * nreq:
            steps += len(sampled)
            tokens += sum(sampled)
    print(f"{name}: {steps} request-steps, {tokens / steps if steps else 0:.2f} tokens per step")

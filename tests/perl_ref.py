"""Runs the vendored Perl mm-th-delay, the reference for mahimahi.mm_th_delay."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from expliccit_env import mahimahi as mm

MM_TH_DELAY = Path(__file__).parents[1] / "third_party" / "mahimahi" / "scripts" / "mm-th-delay"


def run_perl(log: mm.MmLog, tmp: Path) -> str:
    """Run the vendored mm-th-delay; return STDERR without Perl warnings."""
    (tmp / "log").write_text(mm.format_mm_log(log))
    gnuplot = tmp / "bin" / "gnuplot"
    if not gnuplot.exists():
        gnuplot.parent.mkdir(exist_ok=True)
        gnuplot.write_text("#!/bin/sh\ncat > /dev/null\n")
        gnuplot.chmod(0o755)
    env = {**os.environ, "PATH": f"{gnuplot.parent}:{os.environ['PATH']}"}
    r = subprocess.run(
        ["perl", str(MM_TH_DELAY), "500", str(tmp / "log")],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return "".join(
        line
        for line in r.stderr.splitlines(keepends=True)
        if not line.startswith("Use of uninitialized value")
    )

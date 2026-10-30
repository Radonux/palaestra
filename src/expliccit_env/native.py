"""Tier 2: the guest runner as a root process on a native Linux 7.0 host,
in real time (built by scripts/tier2-setup.sh). UNRUN here: this
repository's machines cannot load modules into their own kernel.

Same plan in, same files out as Tier 1. Differences: time is real, the
host's own entropy is used (``seed`` is recorded but cannot make a run
reproducible), and one episode runs at a time (the record ring allows one
reader, so a second concurrent episode fails with EBUSY).
"""

from __future__ import annotations

import contextlib
import os
import platform
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from expliccit_env.uml import GuestRun


@dataclass(frozen=True, slots=True)
class NativeBuild:
    root: Path

    @property
    def runner(self) -> Path:
        return self.root / "expliccit-guest"

    @property
    def modules(self) -> Path:
        return self.root / "modules"

    def build_info(self) -> dict[str, str]:
        lines = (self.root / "BUILD_INFO").read_text().splitlines()
        return dict(line.split("=", 1) for line in lines if "=" in line)

    @staticmethod
    def at(path: Path) -> NativeBuild:
        b = NativeBuild(path.resolve())
        for p in (b.runner, b.modules / "sch_mmlink.ko", b.root / "BUILD_INFO"):
            if not p.exists():
                raise FileNotFoundError(f"{p} missing; run scripts/tier2-setup.sh")
        built_for = b.build_info().get("kernel_release")
        if built_for != platform.release():
            raise RuntimeError(f"modules built for {built_for}, running {platform.release()}")
        return b


def run_native(
    build: NativeBuild, plan_dir: Path, out_dir: Path, *, timeout_s: float = 600.0
) -> GuestRun:
    if os.geteuid() != 0:
        raise PermissionError("Tier 2 episodes need root (namespaces, qdiscs, modules)")
    out_dir.mkdir(parents=True, exist_ok=True)
    console = out_dir / "console.log"
    cmd = [str(build.runner), str(plan_dir.resolve()), str(out_dir.resolve()), str(build.modules)]
    start = time.monotonic()
    with console.open("wb") as log:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc: int | None = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            rc = None
    return GuestRun(rc, time.monotonic() - start, console)

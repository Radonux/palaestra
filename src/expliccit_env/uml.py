"""Launch one guest: a UML kernel in time-travel=inf-cpu mode.

The guest has no wall-clock bound of its own (docs/m0-report.md, F1), so
every run gets a timeout and the whole UML process group is killed on
expiry. UML keeps guest memory in an unlinked file under ``TMPDIR``.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class UmlBuild:
    """A directory produced by scripts/build-uml.sh and build-guest.sh."""

    root: Path

    @property
    def linux(self) -> Path:
        return self.root / "linux"

    @property
    def rootfs(self) -> Path:
        return self.root / "rootfs"

    def build_info(self) -> dict[str, str]:
        lines = (self.root / "BUILD_INFO").read_text().splitlines()
        return dict(line.split("=", 1) for line in lines if "=" in line)

    @staticmethod
    def at(path: Path) -> UmlBuild:
        b = UmlBuild(path.resolve())
        for p in (b.linux, b.rootfs / "init", b.root / "BUILD_INFO"):
            if not p.exists():
                raise FileNotFoundError(f"{p} missing; run scripts/build-uml.sh and build-guest.sh")
        return b


@dataclass(frozen=True, slots=True)
class GuestRun:
    returncode: int | None  # None: killed at the timeout
    wall_s: float
    console: Path


def _cmdline_path(p: Path) -> str:
    s = str(p.resolve())
    if any(c.isspace() for c in s):
        raise ValueError(f"{s}: kernel command-line paths cannot contain whitespace")
    return s


# Guest wall clock at boot. Fixed so nothing in a run depends on host time;
# the episode itself only uses CLOCK_MONOTONIC.
TIME_TRAVEL_START_NS = 1_700_000_000_000_000_000


def run_guest(
    build: UmlBuild,
    plan_dir: Path,
    out_dir: Path,
    *,
    seed: int,
    mem_mib: int = 512,
    timeout_s: float = 600.0,
    tmpdir: Path | None = None,
) -> GuestRun:
    """Boot the guest on ``plan_dir``; its files land in ``out_dir``.

    With the seeded kernel (CONFIG_UML_SEEDED_RANDOM) the run is a pure
    function of the build, the plan and ``seed``.
    """
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be a u64")
    out_dir.mkdir(parents=True, exist_ok=True)
    console = out_dir / "console.log"
    cmd = [
        str(build.linux),
        f"mem={mem_mib}M",
        "time-travel=inf-cpu",
        f"time-travel-start={TIME_TRAVEL_START_NS}",
        f"um_seed={seed}",
        "rootfstype=hostfs",
        f"rootflags={_cmdline_path(build.rootfs)}",
        "ro",
        "init=/init",
        "con=null",
        "con0=null,fd:1",
        f"expl_in={_cmdline_path(plan_dir)}",
        f"expl_out={_cmdline_path(out_dir)}",
    ]
    env = dict(os.environ)
    if tmpdir is not None:
        env["TMPDIR"] = str(tmpdir)
    start = time.monotonic()
    with console.open("wb") as log:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            rc: int | None = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            rc = None
        finally:
            # UML helper processes share the session; never leave them behind.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
    return GuestRun(rc, time.monotonic() - start, console)

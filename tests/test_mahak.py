"""The Mahak driver port (mahak.py): grid laws, and equality with Mahak's own
mahak_online.py run on the same synthetic oracle."""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
from dataclasses import astuple
from pathlib import Path

import numpy as np
import pytest
import scipy

from expliccit_env.mahak import Failure, MahakPoint, MahakSpace, explore
from tests.mahak_ref import synthetic

# run_mahak.sh's search space.
SPACE = ("2,200,4", "5,50,5", "2,20,5", "0.1,2,0.1")
# The last scikit-learn that Mahak's vendored modAL runs on (it passes
# check_X_y's force_all_finite, removed in 1.8).
REFERENCE_SKLEARN = "scikit-learn==1.7.2"
REFERENCE_PANDAS = "pandas==2.3.3"


def oracle(p: MahakPoint) -> float:
    return synthetic(p.buffer_pkts, p.mrtt_ms, p.bw_mbps, p.step)


def test_grid_follows_mahaks_arange() -> None:
    grid = MahakSpace.parse(*SPACE).grid()
    # np.arange(min, max + step, step) passes max when step does not divide
    # the range: buffer reaches 202 and bandwidth 22.
    assert grid.shape == (51 * 10 * 5 * 20, 4)
    assert grid[:, 0].max() == 202 and grid[:, 2].max() == 22
    assert grid[1].tolist() == [2, 5, 2, 0.2]  # the step axis varies fastest
    # A float step can overshoot too: Mahak's own help example, 0.1,4,0.2,
    # ends at 4.10 (rounded to two decimals, as Mahak formats it).
    change = MahakSpace.parse("2,2,1", "5,5,1", "2,2,1", "0.1,4,0.2").grid()[:, 3]
    assert change.tolist()[-2:] == [3.9, 4.1]


def test_failed_queries_never_become_labels() -> None:
    space = MahakSpace.parse("10,20,10", "5,10,5", "5,5,1", "0.5,1,0.5")

    def flaky(p: MahakPoint) -> float | Failure:
        return Failure("boom") if p.buffer_pkts == 10 else float("nan") if p.mrtt_ms == 10 else 1.0

    m = explore(space, flaky, budget=20, seed=0)
    ok = [q for q in m.queries if not isinstance(q.result, Failure)]
    assert ok and all(q.result == 1.0 for q in ok)
    bad = {q.index for q in m.queries if isinstance(q.result, Failure)}
    assert len(bad) == sum(isinstance(q.result, Failure) for q in m.queries)  # never retried
    assert np.allclose(m.mean, 1.0)


@pytest.mark.skipif(
    "EXPLICCIT_MAHAK_DIR" not in os.environ or shutil.which("uv") is None,
    reason="set EXPLICCIT_MAHAK_DIR to a Mahak checkout at 5077583; needs uv",
)
def test_driver_matches_mahak_online(tmp_path: Path) -> None:
    budget, seed = 15, 3
    args = ["-cc", "cubic", "-budget", str(budget), "-metric", "first_power"]
    args += ["-buffer", SPACE[0], "-mRTT", SPACE[1], "-bw", SPACE[2], "-change", SPACE[3]]
    ref = Path(__file__).with_name("mahak_ref.py")
    with_ = [REFERENCE_SKLEARN, REFERENCE_PANDAS, f"numpy=={np.__version__}"]
    with_ += [f"scipy=={scipy.__version__}"]
    subprocess.run(
        ["uv", "run", "--no-project", "--python", platform.python_version()]
        + [a for w in with_ for a in ("--with", w)]
        + ["python", "-I", str(ref), os.environ["EXPLICCIT_MAHAK_DIR"], str(tmp_path), str(seed)]
        + args,
        check=True,
    )
    calls = [tuple(c) for c in json.loads((tmp_path / "calls.json").read_text())]
    with (tmp_path / "predictions" / "Mahak_cubic.csv").open() as f:
        theirs = list(csv.reader(f))

    m = explore(MahakSpace.parse(*SPACE), oracle, budget=budget, seed=seed)
    assert [astuple(q.point) for q in m.queries] == calls
    assert theirs[0] == ["", "buffer_size", "mRTT", "BW", "bw_step", "first_power"]
    their_grid = np.array([[float(x) for x in row[1:5]] for row in theirs[1:]])
    assert np.array_equal(their_grid, m.grid)
    their_mean = np.array([float(row[5]) for row in theirs[1:]])
    np.testing.assert_allclose(m.mean, their_mean, rtol=1e-9, atol=1e-12)

"""Runs Mahak's own ``mahak_online.py``, unmodified, with a synthetic oracle.

Executed in a separate environment that Mahak's vendored modAL still runs in
(scikit-learn before 1.8): ``python -I mahak_ref.py MAHAK_DIR OUT SEED ARGS...``.
It replaces ``helper.oracle_query`` with ``synthetic``, seeds the global
numpy RNG as Mahak would be seeded, and writes ``calls.json`` (every oracle
query, in order) next to Mahak's ``predictions/Mahak_<cc>.csv``. Imports
nothing from expliccit_env, so tests can also import ``synthetic`` from it.
"""

from __future__ import annotations

import json
import os
import runpy
import sys

import numpy as np


def synthetic(buffer: int, mrtt: int, bw: int, step: float) -> float:
    """A cheap, smooth stand-in for a power map: best when the buffer holds
    about one bandwidth-delay product, worse as the capacity step grows."""
    bdp_pkts = bw * 1e6 / 8 / 1500 * 2 * mrtt / 1e3
    fit = np.exp(-(np.log(buffer / bdp_pkts) ** 2) / 2)
    return float(fit / (1 + abs(np.log(step))))


def main(mahak_dir: str, out: str, seed: int, args: list[str]) -> None:
    sys.path.insert(0, mahak_dir)
    import helper  # noqa: PLC0415 (Mahak's module, importable only now)

    calls: list[tuple[int, int, int, float]] = []

    def oracle(x: np.ndarray, metric: str, cc: str) -> np.ndarray:
        del metric, cc
        point = (int(x[0]), int(x[1]), int(x[2]), float(x[3]))
        calls.append(point)
        return np.array([synthetic(*point)])

    helper.oracle_query = oracle
    np.random.seed(seed)  # noqa: NPY002 (Mahak draws from the global RNG)
    os.chdir(out)
    os.makedirs("predictions", exist_ok=True)
    sys.argv = ["mahak_online.py", *args]
    runpy.run_path(os.path.join(mahak_dir, "mahak_online.py"), run_name="__main__")
    with open("calls.json", "w") as f:
        json.dump(calls, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:])

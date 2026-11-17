"""Where net1's capacity belief is too high, on links it never trained on.

Reads stored steps (Sage's observations and the labels) of a net1 training
set, keeps the rows on validation and test links (split by link, seed 1,
15% each, as experiments/net1_train.py) after the first 3 s, and compares
net1's capacity belief with the truth. The mean is what the window target
reads (graph.WindowTarget); the stated spread is the standard deviation of
log2 capacity under the belief (graph.SpreadTarget).

With --less-udp the truth is the capacity less the constant-rate UDP
traffic (data.less_udp), what a net1 trained with that label states.

Writes OUT/summary.md with three tables:
- on clean links (no cross traffic, no random loss), by trace shape and,
  for a link that steps between two levels, the level it is at: the
  belief's mean, median and 10% quantile over the truth (medians over
  rows), the share of rows whose mean is above 1.1 times the truth, and
  the mean probability net1 puts above 1.3 times the truth;
- at the lower level of a stepping clean link, the same by time since the
  step down and whether the link was busy (busy fraction at least 0.95);
- for every condition, log2(mean / truth) by stated spread.

Usage: python experiments/capacity_belief.py OUT NET1.pt DATA... [--stride 12] [--less-udp]
"""

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from expliccit_env.train import data, fit

SPREAD_EDGES = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 99.0)
SINCE_EDGES = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
HEAD = (
    "| rows | n | mean / truth | median / truth | 10% quantile / truth"
    " | mean > 1.1 truth | P(above 1.3 truth) |\n|---|---|---|---|---|---|---|"
)
type F = npt.NDArray[np.float64]
type B = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Rows:
    """Per held-out row: the truth, net1's belief in it, and the row's setting."""

    truth: F
    mean: F
    median: F
    q10: F
    above: F  # probability above 1.3 times the truth
    spread: F
    busy: B
    condition: npt.NDArray[np.str_]
    shape: npt.NDArray[np.str_]  # constant, lower level, higher level
    since: F  # seconds since the last step, -1 before the first


def condition(spec: dict[str, Any]) -> str:
    cross = spec.get("cross") or []
    if any(c.get("kind") == "tcp" for c in cross):
        return "cross TCP"
    if cross:
        return "UDP"
    loss = spec["path"]["loss"]
    return "random loss" if loss["downlink"] or loss["uplink"] else "clean"


def load(net1_path: Path, dirs: list[Path], stride: int, less_udp: bool) -> Rows:
    net1, _ = fit.load_net1(net1_path)
    steps = data.load(dirs, seed=1, val=0.15, test=0.15, stride=stride, split_by="link")
    spec = data.specs(dirs)
    if less_udp:
        steps = data.less_udp(steps, spec)
    idx = np.flatnonzero(np.isin(steps.split, ["val", "test"]) & (steps.t_ms >= 3000))
    eid = np.array(steps.episode_ids)[steps.episode[idx]]
    k = net1.lcfs.index("capacity_bps")
    truth = steps.labels["capacity_bps"][idx].astype(np.float64) / 1e6
    obs = steps.obs[idx]
    dist = net1.distributions(obs)[:, k, :].astype(np.float64)
    bounds = net1.kind.bounds.numpy()[k]
    q = fit.quantiles(dist[:, None, :].astype(np.float32), bounds[None, :], (0.1, 0.5))[:, 0, :]
    log_mid = np.log2(np.maximum(net1.kind.mid[k].numpy(), 1e-3))
    mu = (dist * log_mid).sum(axis=1)
    shape = np.empty(idx.size, "<U16")
    since = np.full(idx.size, -1.0)
    for j, e in enumerate(eid):
        trace = spec[e]["path"]["trace"]
        levels = trace.get("levels_mbps", [0.0])
        if len(set(levels)) == 1:
            shape[j] = "constant"
            continue
        lo, hi = min(levels), max(levels)
        shape[j] = "lower level" if abs(truth[j] - lo) < abs(truth[j] - hi) else "higher level"
        t = float(steps.t_ms[idx[j]])
        if t >= trace["level_ms"]:
            since[j] = (t % trace["level_ms"]) / 1000
    return Rows(
        truth=truth,
        mean=net1.predict(obs)[:, k].astype(np.float64),
        median=q[:, 1].astype(np.float64),
        q10=q[:, 0].astype(np.float64),
        above=(dist * (bounds[None, 1:] > 1.3 * truth[:, None])).sum(axis=1),
        spread=np.sqrt(np.maximum((dist * (log_mid - mu[:, None]) ** 2).sum(axis=1), 0.0)),
        busy=steps.labels["busy_fraction"][idx] >= 0.95,
        condition=np.array([condition(spec[e]) for e in eid]),
        shape=shape,
        since=since,
    )


def row(r: Rows, m: B, label: str) -> str:
    x = r.mean[m] / r.truth[m]
    return (
        f"| {label} | {m.sum()} | {np.median(x):.2f} | {np.median(r.median[m] / r.truth[m]):.2f}"
        f" | {np.median(r.q10[m] / r.truth[m]):.2f} | {np.mean(x > 1.1):.2f}"
        f" | {r.above[m].mean():.2f} |"
    )


def report(r: Rows) -> list[str]:
    clean = r.condition == "clean"
    lines = ["Clean links:", "", HEAD]
    lines += [
        row(r, clean & (r.shape == s), s) for s in ("constant", "lower level", "higher level")
    ]
    lines += ["", "Lower level of a stepping clean link, by time since the step down:", "", HEAD]
    low = clean & (r.shape == "lower level")
    for lo, hi in itertools.pairwise(SINCE_EDGES):
        for b, name in ((r.busy, "busy"), (~r.busy, "not busy")):
            m = low & (r.since >= lo) & (r.since < hi) & b
            if m.sum() >= 200:
                lines.append(row(r, m, f"{lo:g} to {hi:g} s, {name}"))
    over = np.log2(r.mean / r.truth)
    names = [f"{lo:g} to {hi:g}" for lo, hi in itertools.pairwise(SPREAD_EDGES)]
    lines += ["", "Median log2(mean / truth) by stated spread (rows in brackets):", ""]
    lines += ["| condition | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for c in ("clean", "random loss", "UDP", "cross TCP"):
        cells = []
        for lo, hi in itertools.pairwise(SPREAD_EDGES):
            m = (r.condition == c) & (r.spread >= lo) & (r.spread < hi)
            cells.append(f"{np.median(over[m]):+.2f} ({m.sum()})" if m.sum() >= 100 else "")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("net1", type=Path)
    p.add_argument("data", type=Path, nargs="+")
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--less-udp", action="store_true", help="truth: capacity less constant-rate UDP")
    a = p.parse_args()
    truth = "capacity less constant-rate UDP" if a.less_udp else "capacity"
    lines = [
        f"net1 `{a.net1}` against the {truth}, held-out links, after 3 s, every {a.stride}th step.",
        "",
    ]
    lines += report(load(a.net1, a.data, a.stride, a.less_udp))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

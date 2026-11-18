"""Do net1's beliefs keep what tells copies from a stock competitor?

Round 18's copies belief q mistakes Cubic for copies, so a net2 that
cooperates with copies yields to Cubic (exp-net2-eval-v6-coop-176). This
asks whether the information is there to be had: a classifier of the other
traffic on the link (none, copies of the flow, a stock TCP flow) is
trained per step, once on the raw measurements net1 reads (its input
scaler) and once on net1's beliefs (its bin probabilities), and scored on
links it never saw (folds by link). If the raw measurements separate
copies from a stock flow much better than the beliefs do, the LCFs omit
what the decision needs (H1's failure mode); if both fail alike, the
measurements of one step do not carry it.

Usage: python experiments/competitor_info.py OUT NET1.pt
Reads every c-* campaign under OUT whose steps tables keep the
observations (the observed flow one controller throughout); writes
OUT/summary.md.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import torch

from expliccit_env.policy import OBS_DIM
from expliccit_env.train import fit, score
from expliccit_env.train.traffic import CLASSES, fit_head

type F32 = npt.NDArray[np.float32]
SETTLE_S = 5.0
STRIDE = 10  # every 10th step (100 ms): neighbours carry little more
FOLDS = 3


def traffic(condition: str) -> int:
    """The class of an episode's other traffic, from its condition label:
    copies when it names the flow itself, stock when another TCP flow,
    none otherwise (alone, random loss, or only UDP)."""
    parts = [p for p in condition.split(", ") if " from " in p]
    if any(p.startswith("itself") for p in parts):
        return CLASSES.index("copies")
    if any(not p.startswith("UDP") for p in parts):
        return CLASSES.index("stock")
    return CLASSES.index("none")


def senders(condition: str) -> str:
    """The other senders an episode's condition names (``cubic``,
    ``itself``, ...), or the condition itself when it names none."""
    parts = {p.split(" from ")[0] for p in condition.split(", ") if " from " in p}
    return ", ".join(sorted(parts)) or condition


def load(
    out: Path, net1: fit.Net1
) -> tuple[F32, F32, npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.str_], F32]:
    """Per kept step: the scaled measurements, net1's beliefs, the class,
    the link (an index), the other senders and the time since the flow's
    first step (s)."""
    raw, bel, cls, link, who, age = [], [], [], [], [], []
    links: dict[str, int] = {}
    # a shard planned no episodes writes no tables (more shards than episodes)
    for shard in sorted(e.parent for e in out.glob("c-*/shards/*/episodes.parquet")):
        eps = pq.read_table(
            shard / "episodes.parquet", columns=["episode_id", "spec_json", "status"]
        )
        meta = {
            e["episode_id"]: (
                score.condition(e["spec_json"]),
                links.setdefault(score.link_key(e["spec_json"]), len(links)),
            )
            for e in eps.to_pylist()
            if e["status"] == "ok"
        }
        t = pq.read_table(shard / "steps.parquet", columns=["episode_id", "t_ns", "obs"])
        ids = np.asarray(t["episode_id"].to_pylist())
        ts = np.asarray(t["t_ns"].to_numpy(), np.float64) / 1e9
        obs = np.asarray(t["obs"].combine_chunks().flatten(), np.float32).reshape(-1, OBS_DIM)
        for eid in np.unique(ids):
            if eid not in meta:
                continue
            rows = np.flatnonzero(ids == eid)
            rows = rows[ts[rows] >= ts[rows].min() + SETTLE_S][::STRIDE]
            if rows.size == 0:
                continue
            with torch.no_grad():
                raw.append(net1.scaler(torch.from_numpy(obs[rows])).numpy())
            bel.append(net1.beliefs(obs[rows]))
            cls.append(np.full(rows.size, traffic(meta[eid][0])))
            link.append(np.full(rows.size, meta[eid][1]))
            who.append(np.full(rows.size, senders(meta[eid][0])))
            age.append((ts[rows] - ts[ids == eid].min()).astype(np.float32))
    return (
        np.concatenate(raw),
        np.concatenate(bel),
        np.concatenate(cls).astype(np.int64),
        np.concatenate(link).astype(np.int64),
        np.concatenate(who),
        np.concatenate(age),
    )


def balanced_accuracy(pred: npt.NDArray[np.int64], y: npt.NDArray[np.int64]) -> float:
    present = [c for c in range(len(CLASSES)) if (y == c).any()]
    return float(np.mean([(pred[y == c] == c).mean() for c in present]))


def held_out(
    x: F32, y: npt.NDArray[np.int64], link: npt.NDArray[np.int64], seed: int = 0
) -> dict[str, float]:
    """Balanced accuracy of a small MLP over folds by link, and among the
    steps with other TCP traffic, the share of copies called stock and of
    stock called copies."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.unique(link))
    fold = {int(k): i % FOLDS for i, k in enumerate(order)}
    f = np.array([fold[int(k)] for k in link])
    pred = np.zeros_like(y)
    for k in range(FOLDS):
        tr, te = f != k, f == k
        head = fit_head(x[tr], y[tr], seed=seed)
        with torch.no_grad():
            pred[te] = head(torch.from_numpy(x[te])).argmax(1).numpy()
    copies, stock = CLASSES.index("copies"), CLASSES.index("stock")
    return {
        "balanced_accuracy": balanced_accuracy(pred, y),
        "copies_called_stock": _share(pred[y == copies] == stock),
        "stock_called_copies": _share(pred[y == stock] == copies),
    }


def _share(hits: npt.NDArray[np.bool_]) -> float:
    """The share of true entries, NaN when there are none."""
    return float(hits.mean()) if hits.size else float("nan")


def main(out: Path, net1_path: Path) -> None:
    net1, _ = fit.load_net1(net1_path)
    raw, bel, y, link, _, _ = load(out, net1)
    counts = defaultdict(int)
    for c in y.tolist():
        counts[CLASSES[c]] += 1
    rows = {"raw measurements": held_out(raw, y, link), "net1's beliefs": held_out(bel, y, link)}
    lines = [
        "The other traffic on the link (none, copies, a stock TCP flow), classified per step on",
        f"links held out ({FOLDS} folds by link, {len(np.unique(link))} links; steps:"
        f" {', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}).",
        "",
        "| input | balanced accuracy | copies called stock | stock called copies |",
        "|---|---|---|---|",
    ]
    for name, m in rows.items():
        lines.append(
            f"| {name} | {m['balanced_accuracy']:.3f} | {m['copies_called_stock']:.3f}"
            f" | {m['stock_called_copies']:.3f} |"
        )
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

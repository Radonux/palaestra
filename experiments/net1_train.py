"""Train one net1 on campaign data and report it, as net1 v4 and v5 were trained.

Usage: python experiments/net1_train.py OUT --spec SPEC.json --data DIR... \
           [--set KEY=VALUE ...] [--ood NAME=DIR ...] [--compare NET1.pt] [--seed 0]
           [--stride 12] [--eval NET1.pt] [--compare-expliccit MODELS_DIR]

Rows are every 12th step (--stride; the stored data keeps every 6th, and
a runner's 16 GB held net1 v5's 7.2 million rows but not net1 v6's 12.6
million), split by link (seed 1, 15% validation, 15% test).
BBRv3, Westwood and threshold-0.3 are held out of training; their rows on
validation and test links are reported apart. Each --ood dataset is
reported whole. --compare reports another net1 on the same views, for a
before and after, and --compare-expliccit does the same for expliCCit's
released net1 (third_party/expliccit-models). --eval reports a trained net1 instead of training one
(net1 v6 against v5 on v5's own data). Writes OUT/net1.pt, OUT/report.json and OUT/summary.md
(R^2 and absolute error per LCF and view, and for a distribution net1
its 90% coverage, CRPS skill and log loss; selection on the validation
rows, never on the test rows).
"""

import argparse
import dataclasses
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from expliccit_env.train import data, fit
from expliccit_env.train.spec import Net1Spec

ROOT = Path(__file__).resolve().parents[1]
STRIDE = 12


def note(stage: str) -> None:
    """One line per stage with the peak memory so far: a runner that runs
    out of memory is killed without a message, and this says where."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"{time.strftime('%H:%M:%S')} {stage}: peak {peak:.1f} GB", flush=True)


def held() -> dict[str, str]:
    t03 = hashlib.sha256((ROOT / "examples/policies/threshold-0.3.onnx").read_bytes()).hexdigest()
    return {"bbr3": "bbr3", "westwood": "westwood", f"agent:{t03[:12]}": "threshold-0.3"}


def controllers(dirs: list[Path], steps: data.Steps) -> np.ndarray[Any, np.dtype[np.str_]]:
    of: dict[str, str] = {}
    for d in dirs:
        for f in d.rglob("episodes.parquet"):
            t = pq.read_table(f, columns=["episode_id", "controller"])
            of |= dict(zip(t["episode_id"].to_pylist(), t["controller"].to_pylist(), strict=True))
    return np.array([of[e] for e in steps.episode_ids])[steps.episode]


def summary(rep: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Per LCF, the mean's R^2 and absolute error; for a distribution net1
    also its coverage, CRPS skill and log loss."""
    out: dict[str, dict[str, float]] = {}
    for k, v in rep.items():
        if not (isinstance(v, dict) and "net1" in v):
            continue
        out[k] = {
            "r2": v["net1"].get("r2", float("nan")),
            "mae": v["net1"].get("mae", float("nan")),
        }
        if "distribution" in v:
            d = v["distribution"]
            out[k] |= {
                "c90": d["coverage"]["0.9"],
                "skill": d["crps_skill"],
                "log_loss": d["log_loss"],
            }
    return out


def train(
    a: argparse.Namespace, spec: Net1Spec, others: dict[str, fit.Net1]
) -> tuple[fit.Net1, dict[str, Any]]:
    """Fit on the training rows and report on the split's views; the data is
    freed when this returns, before the held-out sets load."""
    t0 = time.time()
    steps = data.load(a.data, seed=1, val=0.15, test=0.15, stride=a.stride, split_by="link")
    ctl = controllers(a.data, steps)
    hold = held()
    is_held = np.isin(ctl, list(hold))
    base = np.where(is_held, "hold", steps.split)
    keep = np.where(base == "train", "train", "other")

    def view(rows: np.ndarray[Any, np.dtype[np.bool_]], split: str) -> data.Steps:
        """Rows on ``split`` links become "test"; training rows stay (climatology)."""
        s = np.where(rows & (steps.split == split), "test", keep)
        return dataclasses.replace(steps, split=s.astype("<U5"))

    views = {"val": (~is_held, "val"), "test": (~is_held, "test"), "test_held": (is_held, "test")}
    views |= {f"test_{name}": (ctl == c, "test") for c, name in hold.items()}
    load_s = time.time() - t0
    note(f"loaded {steps.obs.shape[0]} rows")
    if a.eval:
        net1, log = fit.load_net1(a.eval)[0], {}
    else:
        net1, log = fit.fit_net1(spec, dataclasses.replace(steps, split=base), a.seed)
    rows = {k: int((base == k).sum()) for k in ("train", "val", "test", "hold")}
    rep: dict[str, Any] = {"log": log, "rows": rows, "load_s": load_s, "views": {}}
    rep["train_s"] = time.time() - t0 - load_s
    note(f"trained, {rows}")
    for name, (mask, split) in views.items():
        v = view(mask, split)
        rep["views"][name] = {"net1": summary(fit.report_net1(net1, v, "test"))}
        for who, other in others.items():
            rep["views"][name][who] = summary(fit.report_net1(other, v, "test"))
            note(f"reported {name}, {who}")
        note(f"reported {name}")
    return net1, rep


def table(rep: dict[str, Any], lcfs: list[str], title: str) -> str:
    nan = float("nan")  # a single-value net1 states no coverage and no log loss
    lines = [
        title,
        "",
        "| view | model | " + " | ".join(lcfs) + " |",
        "|" + "---|" * (len(lcfs) + 2),
    ]
    for name, v in rep["views"].items():
        for model, s in v.items():
            cells = [
                f"{s[k].get('c90', nan):.2f} / {s[k]['r2']:.2f} / {s[k].get('log_loss', nan):.2f}"
                if k in s
                else ""
                for k in lcfs
            ]
            who = "this" if model == "net1" else model
            lines.append(f"| {name} | {who} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--set", action="append", default=[], help="KEY=VALUE (JSON) over the spec")
    p.add_argument("--ood", action="append", default=[], help="NAME=DIR, reported whole")
    p.add_argument("--compare", type=Path, help="another net1.pt to report on the same views")
    p.add_argument(
        "--compare-expliccit", type=Path, help="expliCCit's released models, reported the same way"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stride", type=int, default=STRIDE, help="keep every Nth step")
    p.add_argument("--eval", type=Path, help="a trained net1.pt to report instead of training")
    a = p.parse_args()
    raw = json.loads(a.spec.read_text())
    raw |= {k: json.loads(v) for k, v in (x.split("=", 1) for x in a.set)}
    spec = Net1Spec.model_validate(raw)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    others: dict[str, fit.Net1] = {}
    if a.compare:
        others["compare"] = fit.load_net1(a.compare)[0]
    if a.compare_expliccit:
        others["expliccit"] = fit.load_expliccit(a.compare_expliccit)[0]
    net1, rep = train(a, spec, others)
    if not a.eval:
        fit.save_net1(net1, spec, a.out / "net1.pt")
    rep |= {"spec": spec.model_dump(mode="json"), "seed": a.seed}
    for name, d in (x.split("=", 1) for x in a.ood):
        o = data.load([Path(d)], seed=1, val=0.0, test=1.0, stride=a.stride, split_by="link")
        rep["views"][f"ood_{name}"] = {"net1": summary(fit.report_net1(net1, o, "test"))}
        for who, other in others.items():
            rep["views"][f"ood_{name}"][who] = summary(fit.report_net1(other, o, "test"))
        note(f"reported ood_{name}, {o.obs.shape[0]} rows")
    (a.out / "report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    title = (
        f"net1, dropout {raw.get('dropout')}, width {raw.get('width')}, seed {a.seed}: 90% coverage"
        f" / R^2 / log loss (nats) per LCF; {rep['rows']['train']} training rows,"
        f" {rep['train_s']:.0f} s to train."
    )
    (a.out / "summary.md").write_text(table(rep, list(spec.lcfs), title))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

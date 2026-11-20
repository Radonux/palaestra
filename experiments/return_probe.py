"""How well does what an actor reads predict the return its critic fits?

Usage: python experiments/return_probe.py OUT DATA NET1.pt NAME=DIR... [A+B...]
       python experiments/return_probe.py --summary OUT

DIR is a net2 training output (net2.pt, train_spec.json); every DIR must
share one seed, split and reward. For each, the input its actor and critic
read, built as training built it (fit.net2_view): net1's estimates, the true
LCFs, or for a free encoder the scaled measurements through the run's own
trained encoder, which also adds "raw", the scaled measurements themselves.
A+B joins two inputs side by side. The target is what the critic fits
before bootstrapping: the discounted sum of the next n rewards after the
logged action (fit.transitions). A map from (input, action) to that sum is
fitted on the training links of the split and scored by R^2 on its
validation links, on every 10th transition: least squares with a bias, and a
network with two hidden layers of 128. Writes OUT/probe.json and probe.md;
--summary gathers OUT/*/probe.json into OUT/summary.md.
"""

import json
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from latent_probe import _r2
from torch import nn

from expliccit_env.train import data, fit
from expliccit_env.train.spec import TrainSpec

type F32 = npt.NDArray[np.float32]
STRIDE = 10
MLP_STEPS = 3000
BATCH = 4096
EVAL = 65536


@torch.no_grad()
def encode(f: nn.Module, x: F32) -> F32:
    return np.concatenate(
        [f(torch.from_numpy(x[j : j + EVAL])).numpy() for j in range(0, len(x), EVAL)]
    )


def linear_r2(x: F32, y: F32, tr: npt.NDArray[np.bool_], va: npt.NDArray[np.bool_]) -> float:
    def design(rows: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
        return np.c_[x[rows].astype(np.float64), np.ones(int(rows.sum()))]

    w, *_ = np.linalg.lstsq(design(tr), y[tr].astype(np.float64), rcond=None)
    return _r2(design(va) @ w, y[va].astype(np.float64))


@torch.enable_grad()
def network_r2(x: F32, y: F32, tr: npt.NDArray[np.bool_], va: npt.NDArray[np.bool_]) -> float:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    ym, ys = float(y[tr].mean()), float(y[tr].std()) + 1e-6
    xt = torch.from_numpy(((x - mu) / sd).astype(np.float32))
    yt = torch.from_numpy(((y - ym) / ys).astype(np.float32))
    net = nn.Sequential(
        nn.Linear(x.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rows = np.flatnonzero(tr)
    for _ in range(MLP_STEPS):
        b = torch.from_numpy(rng.choice(rows, BATCH))
        loss = ((net(xt[b]).squeeze(1) - yt[b]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = net(xt[torch.from_numpy(np.flatnonzero(va))]).squeeze(1).numpy() * ys + ym
    return _r2(p.astype(np.float64), y[va].astype(np.float64))


def main(out: Path, data_dir: Path, net1_path: Path, args: list[str]) -> None:
    net1, _ = fit.load_net1(net1_path)
    runs = {n: Path(d) for n, d in (a.split("=", 1) for a in args if "=" in a)}
    joins = [a.split("+") for a in args if "=" not in a]
    specs = {
        n: TrainSpec.model_validate_json((d / "train_spec.json").read_text())
        for n, d in runs.items()
    }
    splits = {
        (t.seed, t.val_fraction, t.test_fraction, t.split_by, t.step_stride) for t in specs.values()
    }
    if len(splits) != 1:
        raise SystemExit("the runs split the data differently")
    s = next(iter(specs.values()))
    steps = data.load(
        [data_dir],
        seed=s.seed,
        val=s.val_fraction,
        test=s.test_fraction,
        stride=s.step_stride,
        pairs=True,
        split_by=s.split_by,
    )
    probe(out, steps, net1, {n: (specs[n], d) for n, d in runs.items()}, joins)


def probe(
    out: Path,
    steps: data.Steps,
    net1: fit.Net1,
    runs: dict[str, tuple[TrainSpec, Path]],
    joins: list[list[str]],
) -> None:
    returns = {
        (t.net2.reward, t.net2.n_step, t.net2.discount, t.net2.action, t.net2.reward_scale)
        for t, _ in runs.values()
    }
    if len(returns) != 1:
        raise SystemExit("the runs differ in their return")
    s = next(iter(runs.values()))[0]
    if s.net2.action != "multiplier":
        raise SystemExit("the logged action is net2's only for the multiplier action")
    inputs: dict[str, F32] = {}
    for name, (spec, d) in runs.items():
        n2 = spec.net2
        states, _ = fit.net2_view(net1, n2, steps)
        if n2.encoder == "free":
            f = fit.free_encoder(net1, s.seed)
            state = torch.load(d / "net2.pt", map_location="cpu", weights_only=True)
            f.load_state_dict({k[2:]: v for k, v in state.items() if k.startswith("0.")})
            inputs["raw"] = states
            inputs[name] = encode(f.eval(), states)
        else:
            inputs[name] = states
    for parts in joins:
        inputs["+".join(parts)] = np.concatenate([inputs[p] for p in parts], axis=1)
    # Every run shares the return, so one run's transitions index them all.
    actions = fit.logged_actions(steps, s.net2, np.full(len(steps.obs), np.nan, np.float32))
    tr = fit.transitions(steps, torch.zeros(len(steps.obs), 1), actions, s.net2, "train")[0]
    va = fit.transitions(steps, torch.zeros(len(steps.obs), 1), actions, s.net2, "val")[0]
    i = np.concatenate([tr.i.numpy()[::STRIDE], va.i.numpy()[::STRIDE]])
    a = np.concatenate([tr.a.numpy()[::STRIDE], va.a.numpy()[::STRIDE]])
    y = np.concatenate([tr.r.numpy()[::STRIDE], va.r.numpy()[::STRIDE]])
    is_tr = np.arange(len(i)) < len(tr.i.numpy()[::STRIDE])
    result: dict[str, dict[str, float]] = {}
    for name, z in inputs.items():
        x = np.c_[z[i], a]
        ok = np.isfinite(x).all(1)
        result[name] = {
            "width": z.shape[1],
            "dropped_rows": int((~ok).sum()),
            "linear": linear_r2(x, y, is_tr & ok, ~is_tr & ok),
            "network": network_r2(x, y, is_tr & ok, ~is_tr & ok),
        }
    result["_return"] = {
        "train": int(is_tr.sum()),
        "val": int((~is_tr).sum()),
        "mean": float(y.mean()),
        "std": float(y.std()),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps(result, indent=1) + "\n")
    lines = [
        f"Seed {s.seed}: validation R^2 of a map from (input, logged action) to the"
        f" {s.net2.n_step}-step discounted return, fitted on training links, every"
        f" {STRIDE}th transition ({result['_return']['train']} train,"
        f" {result['_return']['val']} validation).",
        "",
        "| input | width | linear | network |",
        "|---|---|---|---|",
    ]
    for name, r in result.items():
        if not name.startswith("_"):
            lines.append(f"| {name} | {r['width']} | {r['linear']:.3f} | {r['network']:.3f} |")
    (out / "probe.md").write_text("\n".join(lines) + "\n")


def summary(out: Path) -> None:
    runs = sorted(out.glob("*/probe.json"))
    rs = [json.loads(p.read_text()) for p in runs]
    names = [n for n in rs[0] if not n.startswith("_")]
    lines = [
        "Validation R^2 of a map from (input, logged action) to the critic's target, the"
        " discounted return over the next steps, per training seed: mean over seeds"
        " (min to max).",
        "",
        "| input | linear | network |",
        "|---|---|---|",
    ]
    for n in names:
        cells = []
        for kind in ("linear", "network"):
            v = np.array([r[n][kind] for r in rs])
            cells.append(f"{v.mean():.3f} ({v.min():.3f} to {v.max():.3f})")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", *(p.parent.joinpath("probe.md").read_text() for p in runs)]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(Path(sys.argv[2]))
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])

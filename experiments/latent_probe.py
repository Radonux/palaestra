"""How much of net1's LCFs can a map read from the seven numbers a net2's
actor reads?

Usage: python experiments/latent_probe.py OUT DATA NET1.pt NAME=DIR...

DIR is a training output (net2.pt and train_spec.json): its latents are
net1's output for a net2 on net1, or for a free arm its encoder's output
("0.*" of net2.pt). NAME=random is the control, the free encoder's fresh
weights (seed 0) with its batch norms' statistics taken from the data. For
each LCF, a least-squares map with a bias from the latents to the LCF in
net1's target space (its LCF scaler; rows where the label is undefined are
left out) is fitted on the training links of DIR's split and scored by R^2
on its validation links, on every 10th step of each episode; so is a small
network (two hidden layers of 128) fitted to all seven at once, which also
finds what a linear map cannot. Writes OUT/probe.json and OUT/probe.md.
"""

import json
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from expliccit_env.train import data, fit
from expliccit_env.train.spec import TrainSpec

type F32 = npt.NDArray[np.float32]
STRIDE = 10
BATCH = 65536
MLP_STEPS = 3000
BATCH_MLP = 4096


@torch.no_grad()
def latents(f: nn.Module, net1: fit.Net1, obs: F32) -> F32:
    out = np.empty((obs.shape[0], net1.kind.k), np.float32)
    for i in range(0, obs.shape[0], BATCH):
        out[i : i + BATCH] = f(net1.scaler(torch.from_numpy(obs[i : i + BATCH]))).numpy()
    return out


@torch.no_grad()
def random_encoder(net1: fit.Net1, obs: F32) -> nn.Module:
    """Fresh weights; batch norms keep the data's mean statistics."""
    enc = fit.free_encoder(net1, 0)
    for m in enc.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None  # a cumulative mean over the batches below
    for i in range(0, obs.shape[0], BATCH):
        enc(net1.scaler(torch.from_numpy(obs[i : i + BATCH])))
    return enc.eval()


def r2(z: F32, y: F32, tr: npt.NDArray[np.bool_], va: npt.NDArray[np.bool_]) -> float:
    def design(rows: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
        return np.c_[z[rows].astype(np.float64), np.ones(int(rows.sum()))]

    w, *_ = np.linalg.lstsq(design(tr), y[tr].astype(np.float64), rcond=None)
    t = y[va].astype(np.float64)
    return _r2(design(va) @ w, t)


def _r2(p: npt.NDArray[np.float64], t: npt.NDArray[np.float64]) -> float:
    """NaN where the truth does not vary (or is missing) on validation rows."""
    sst = ((t - t.mean()) ** 2).sum() if t.size else 0.0
    return float(1 - ((p - t) ** 2).sum() / sst) if sst > 0 else float("nan")


@torch.enable_grad()
def mlp_r2(
    z: F32,
    y: F32,
    defined: npt.NDArray[np.bool_],
    tr: npt.NDArray[np.bool_],
    va: npt.NDArray[np.bool_],
) -> list[float]:
    """Validation R^2 per LCF of a network from the latents to all LCFs."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    mu, sd = z[tr].mean(0), z[tr].std(0) + 1e-6
    x = torch.from_numpy((z - mu) / sd)
    t, m = torch.from_numpy(np.where(defined, y, 0.0).astype(np.float32)), torch.from_numpy(defined)
    net = nn.Sequential(
        nn.Linear(z.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, y.shape[1]),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rows = np.flatnonzero(tr)
    for _ in range(MLP_STEPS):
        b = torch.from_numpy(rng.choice(rows, BATCH_MLP))
        err = (net(x[b]) - t[b]) ** 2 * m[b]
        loss = err.sum() / m[b].sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = np.concatenate([net(x[i : i + BATCH]).numpy() for i in range(0, len(x), BATCH)])
    out = []
    for k in range(y.shape[1]):
        v = va & defined[:, k]
        out.append(_r2(p[v, k].astype(np.float64), y[v, k].astype(np.float64)))
    return out


def main() -> None:
    out, data_dir, net1_path, *pairs = sys.argv[1:]
    net1, _ = fit.load_net1(Path(net1_path))
    named = [p.partition("=")[::2] for p in pairs]
    specs = {
        n: TrainSpec.model_validate_json((Path(d) / "train_spec.json").read_text())
        for n, d in named
        if d != "random"
    }
    s = next(iter(specs.values()))
    if len({(t.val_fraction, t.test_fraction, t.split_by) for t in specs.values()}) > 1:
        raise SystemExit("the runs split the data differently beyond their seeds")
    seed_of = {n: specs[n].seed if d != "random" else 0 for n, d in named}
    result: dict[str, dict[str, dict[str, float]]] = {}
    for seed in sorted(set(seed_of.values())):
        steps = data.load(
            [Path(data_dir)],
            seed=seed,
            val=s.val_fraction,
            test=s.test_fraction,
            stride=STRIDE,
            split_by=s.split_by,
        )
        y = fit.targets(steps, net1.lcfs)
        defined = np.isfinite(y)
        y = net1.lcf_scaler(torch.from_numpy(y)).numpy()
        tr, va = steps.rows("train"), steps.rows("val")
        for name, d in named:
            if seed_of[name] != seed:
                continue
            if d == "random":
                f = random_encoder(net1, steps.obs[tr])
            elif specs[name].net2.encoder == "free":
                f = fit.free_encoder(net1, seed)
                state = torch.load(Path(d) / "net2.pt", map_location="cpu", weights_only=True)
                f.load_state_dict({k[2:]: v for k, v in state.items() if k.startswith("0.")})
                f.eval()
            else:
                f = net1.net.eval()
            z = latents(f, net1, steps.obs)
            result[name] = {
                "linear": {
                    lcf: r2(z, y[:, k], tr & defined[:, k], va & defined[:, k])
                    for k, lcf in enumerate(net1.lcfs)
                },
                "network": dict(zip(net1.lcfs, mlp_r2(z, y, defined, tr, va), strict=True)),
            }
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "probe.json").write_text(json.dumps(result, indent=1) + "\n")
    lines = [
        "Validation R^2 of a map from the actor's seven inputs to each LCF in net1's target"
        " space, fitted on training links, every 10th step: a linear map with a bias, and a"
        " network with two hidden layers of 128.",
    ]
    for kind in ("linear", "network"):
        lines += [
            "",
            f"| latents ({kind}) | " + " | ".join(net1.lcfs) + " | mean |",
            "|" + "---|" * (len(net1.lcfs) + 2),
        ]
        for name, r in result.items():
            v = list(r[kind].values())
            cells = " | ".join(f"{x:.3f}" for x in v)
            lines.append(f"| {name} | {cells} | {np.mean(v):.3f} |")
    (o / "probe.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

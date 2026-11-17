"""Export several trained net2s of one setting (train/ppo.py) as one
policy on the mean of their outputs (recurrent.EnsembleActor): for gate
students ("gate", "gate3") the mean belief, with the switch sharpened at
--t as in gate_threshold_export.py (and for "gate3", q at --tq and the
copies policy --copies); for students that learn u (PPO, or DAgger "u"),
the mean u. Seeds of one setting misfire on different links, so their
mean should misfire less often.

Usage: python experiments/gate_ensemble_export.py OUT NAME RUN_DIR... [--t 0.5]
       [--tq 0] [--copies rule]
Each RUN_DIR is an `expliccit train ppo` output (ppo_spec.json, net1.pt,
actor.pt) of the same setting but its seed; writes OUT/NAME.onnx.

--truth LCF... takes those LCFs from the guest's live truth instead of
net1 (all probability on the true value's bin), to measure what each
LCF's estimation error costs this net2.

--traffic HEAD.pt gives net1 its traffic head (train/traffic.py) and takes
a gate3 student's q from the head's P(copies) instead of the students;
--traffic-p takes p from its P(stock) too, so that net2 is the rules alone.

--spread K --s0 S0 adds K times net1's stated spread of capacity past S0
to a gate3 student's u, in proportion to its belief that no stock TCP flow
is active, or with --spread-gate copies to its belief in copies
(actors.SpreadHead); K < 0 lowers the window where net1 is unsure.

--net1 NET1.pt runs the actors on another net1 than the one they were
trained on, unchanged (no leakage, H4; repair without retraining net2,
H5). It must give the same outputs in the same bins: the same LCFs, bin
boundaries and input scaler.
"""

import argparse
from pathlib import Path

import torch

from expliccit_env.train import actors, fit, ppo
from expliccit_env.train.export import to_onnx
from expliccit_env.train.recurrent import EnsembleActor, RecurrentActor
from expliccit_env.train.traffic import load_head


def _same_outputs(a: fit.Net1, b: fit.Net1) -> bool:
    """Whether b's output means what a's does: the same LCFs, bin
    boundaries and input scaler (seeds of one net1 training run)."""
    bounds = [getattr(n.net, "bounds", None) for n in (a, b)]
    return (
        a.lcfs == b.lcfs
        and bounds[0] is not None
        and bounds[1] is not None
        and torch.equal(bounds[0], bounds[1])
        and all(
            torch.equal(getattr(a.scaler, k), getattr(b.scaler, k))
            for k in ("cols", "lo", "hi", "center", "spread")
        )
    )


def _specs(runs: list[Path], traffic: Path | None) -> list[ppo.PPOSpec]:
    """Each run's spec, an actor on net1's traffic head reading the head the
    run kept beside its net1.pt."""
    specs = [ppo.PPOSpec.parse((r / "ppo_spec.json").read_text()) for r in runs]
    if specs[0].inputs != "net1+traffic":
        return specs
    if traffic is not None:
        raise SystemExit("--traffic replaces beliefs the actor learned; this actor reads the head")
    return [
        s.model_copy(update={"traffic_head": r / "traffic_head.pt"})
        for s, r in zip(specs, runs, strict=True)
    ]


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("out", type=Path)
    a.add_argument("name")
    a.add_argument("runs", type=Path, nargs="+")
    a.add_argument("--t", type=float, default=0.0, help="gate students' threshold")
    a.add_argument("--tq", type=float)
    a.add_argument("--copies", choices=["coop", "rule"])
    a.add_argument("--net1", type=Path, help="run the actors on this net1 instead")
    a.add_argument("--truth", nargs="+", default=[], help="LCFs to take from the live truth")
    a.add_argument(
        "--traffic", type=Path, help="net1's traffic head (traffic_head.pt): q from it, for gate3"
    )
    a.add_argument("--traffic-p", action="store_true", help="with --traffic, p from the head too")
    a.add_argument("--spread", type=float, help="K: u + K relu(sd - S0) (1 - p), for gate3")
    a.add_argument("--s0", type=float, default=0.15)
    a.add_argument("--spread-gate", choices=["no_stock", "copies"], default="no_stock")
    args = a.parse_args()
    specs = _specs(args.runs, args.traffic)
    free = {"seed": 0, "init_actor": None, "init_critic": None, "traffic_head": None}
    same = {s.model_copy(update=free).model_dump_json() for s in specs}
    if len(same) != 1:
        raise SystemExit("the runs must be of one setting, differing only in seed and start")
    gate = specs[0].student.kind in ("gate", "gate3")
    spec = specs[0].with_gate(args.t, args.tq, args.copies) if gate else specs[0]
    net1, _ = fit.load_net1(args.runs[0] / "net1.pt")
    if args.net1 is not None:
        other, _ = fit.load_net1(args.net1)
        if not _same_outputs(net1, other):
            raise SystemExit(f"{args.net1}: not the same LCFs, bins and input scaler")
        net1 = other
    width = actors.width(net1, spec)
    members: list[RecurrentActor] = []
    for r in args.runs:
        outputs = 2 if spec.student.kind == "gate3" else 1
        actor = RecurrentActor(width, spec.hidden, memory=spec.memory, outputs=outputs)
        actor.load_state_dict(torch.load(r / "actor.pt", weights_only=True))
        members.append(actor)
    args.out.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(args.truth) - set(net1.lcfs))
    if unknown:
        raise SystemExit(f"--truth {unknown}: not among net1's LCFs {list(net1.lcfs)}")
    if args.traffic_p and args.traffic is None:
        raise SystemExit("--traffic-p takes p from the head given by --traffic")
    if args.traffic is not None and spec.student.kind != "gate3":
        raise SystemExit("--traffic replaces a gate3 student's q")
    head = None if args.traffic is None else load_head(args.traffic)
    decide = (
        None if head is None else actors.Head(net1, spec, traffic_q=True, traffic_p=args.traffic_p)
    )
    if args.spread is not None:
        base = decide if decide is not None else actors.Head(net1, spec)
        decide = actors.SpreadHead(base, net1, spec, args.spread, args.s0, gate=args.spread_gate)
    graph = ppo.policy_graph(
        net1, EnsembleActor(members), spec, frozenset(args.truth), traffic=head, head=decide
    )
    (args.out / f"{args.name}.onnx").write_bytes(to_onnx(graph))
    print(f"{args.name}.onnx")


if __name__ == "__main__":
    main()

"""Export the fixed rule on a net1: net2 always answers u, so the next window
is 2^u times net1's believed bandwidth-delay product.

Usage: python experiments/rule_export.py OUT NET1.pt --u -0.25 0 0.25 [--oracle]
           [--spread K ... [--s0 S0]] [--idle K ... [--i0 I0]]

For each u, writes OUT/rule-u{u}.onnx, where net2 reads net1's distributions
pooled to 32 bins (or, for a net1 with one value or an interval per LCF, all
of its output), and OUT/rule-u{u}-actor.pt, the same net2 as an actor (a
starting point for online training). --oracle writes OUT/rule-u{u}-oracle.onnx
instead: the same rule on the true LCFs, read from the guest's live truth.
The actor's last layer is zero and its bias atanh(u / 3), 3 being action_max.
--spread K also writes OUT/rule-u{u}-spread{K}.onnx for each K: the same rule
with its window target raised by K times net1's stated spread of capacity
past S0 (train/models.py SpreadTarget), the arm that reads the beliefs'
uncertainty beside the rule on their means. With --traffic HEAD.pt, net1
also has its traffic head and each spread rule lifts its target only in
proportion to 1 - P(copies) (train/graph.py SpreadTarget), written as
OUT/rule-u{u}-spread{K}-head.onnx. --idle K writes OUT/rule-u{u}-idle{K}.onnx:
the rule with its target raised by K times how idle net1 believes the link
is past I0 (train/models.py IdleTarget), the probing arm on means alone.
--fault LCF*F (or LCF=V) makes every graph's belief in that LCF wrong as a
faulty encoder head would (fit.parse_faults; R2's injected faults).
"""

import argparse
import math
from pathlib import Path

import torch

from expliccit_env.train import fit
from expliccit_env.train.beliefs import Bins
from expliccit_env.train.export import to_onnx
from expliccit_env.train.graph import IdleTarget, SpreadTarget, WindowTarget
from expliccit_env.train.models import ExpliccitActor
from expliccit_env.train.traffic import load_head

ACTION_MAX = 3.0
BINS = 32
HIDDEN = 256


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("net1", type=Path)
    p.add_argument("--u", type=float, nargs="+", required=True)
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--spread", type=float, nargs="*", default=[])
    p.add_argument("--s0", type=float, default=0.2)
    p.add_argument("--traffic", type=Path, help="net1's traffic head: gate the spread on copies")
    p.add_argument("--idle", type=float, nargs="*", default=[])
    p.add_argument("--i0", type=float, default=0.05)
    p.add_argument("--fault", action="append", default=[], help="LCF*FACTOR or LCF=VALUE")
    a = p.parse_args()
    net1, _ = fit.load_net1(a.net1)
    faults = fit.parse_faults(a.fault, net1.lcfs) if a.fault else None
    truth = frozenset(net1.lcfs) if a.oracle else frozenset[str]()
    bins = BINS if isinstance(net1.kind, Bins) else None
    width = fit.input_width(fit.front(net1, input_bins=bins))
    a.out.mkdir(parents=True, exist_ok=True)
    for u in a.u:
        torch.manual_seed(0)  # the hidden layers do not reach the output; same bytes each run
        actor = ExpliccitActor(width, HIDDEN)
        last = actor.net[4]
        assert isinstance(last, torch.nn.Linear)
        with torch.no_grad():
            last.weight.zero_()
            last.bias.fill_(math.atanh(u / ACTION_MAX))
        graph = fit.policy_graph(
            net1,
            actor.eval(),
            ACTION_MAX,
            action="bdp",
            truth=truth,
            input_bins=bins,
            faults=faults,
        )
        name = f"rule-u{u:g}" + ("-oracle" if a.oracle else "")
        (a.out / f"{name}.onnx").write_bytes(to_onnx(graph))
        if not a.oracle:
            torch.save(actor.state_dict(), a.out / f"{name}-actor.pt")
        print(name)
        base = graph.target
        assert isinstance(base, WindowTarget)
        for k in a.idle:
            graph.target = IdleTarget(base, net1.lcfs.index("busy_fraction"), k, a.i0)
            (a.out / f"{name}-idle{k:g}.onnx").write_bytes(to_onnx(graph))
            print(f"{name}-idle{k:g}")
        if a.traffic is not None:
            head = load_head(a.traffic)
            graph.front = fit.front(net1, input_bins=bins, truth=truth, traffic=head)
        for k in a.spread:
            graph.target = SpreadTarget(base, k, a.s0, copies_gated=a.traffic is not None)
            spread = f"{name}-spread{k:g}" + ("" if a.traffic is None else "-head")
            (a.out / f"{spread}.onnx").write_bytes(to_onnx(graph))
            print(spread)


if __name__ == "__main__":
    main()

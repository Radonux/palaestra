"""Write a campaign on net2's own training links: examples/net1_broad.py's
net2-train set (600 links, a pool of 64 conditions, 30 s), whose step data
(exp-data-net2-steps-222) the offline net2s are trained on. Every link is
one the data had and every condition one of its pool; each link runs under
2 of them drawn with the campaign seed, so the pairing differs from the
data's. A one-off: the cost of the LCFs' meaning (experiments/lcf-cost.json)
is scored where both arms were trained, not on held-out links. With --set
net2-train-xl the links and the pool of conditions are new draws from the
same ranges: no link is in the data, and only the simplest conditions (the
flow alone, one Cubic flow from the start) recur. It checks that a result
found where the policies were trained holds where they were not. With
--alone only the pool's conditions without cross traffic are kept (27 of
64 in either set, about half of them with random loss): the flow alone on
a link of set capacity (constant or stepping between two levels), minimum
RTT, buffer and loss, as in expliCCit's own experiments.

Usage: python experiments/comfort.py OUT_DIR --links 0:100 --seed 31
           [--set net2-train-xl] [--alone] --stock bbr3 --policy NAME=PATH ...

Writes OUT_DIR/campaign.json and OUT_DIR/aliases.json, as heldout.py does.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from heldout import controllers
from net1_broad import SETS, draw

SETS_HERE = ("net2-train", "net2-train-xl")


def campaign(
    first: int,
    last: int,
    seed: int,
    ccs: list[dict[str, object]],
    *,
    set_: str = "net2-train",
    alone: bool = False,
) -> dict[str, object]:
    links, conditions = draw(set_)
    if not 0 <= first < last <= len(links):
        raise ValueError(f"links {first}:{last} are not within the set's {len(links)}")
    if alone:
        conditions = [c for c in conditions if not c["cross"]]
    name = f"comfort-{first}-{last}" if set_ == "net2-train" else f"{set_}-{first}-{last}"
    return {
        "name": name + ("-alone" if alone else ""),
        "ccs": ccs,
        "sweep": {"kind": "paths", "paths": links[first:last]},
        "conditions": conditions,
        "conditions_per_point": 2,
        "duration_ms": SETS[set_][2] * 1000,
        "seed": seed,
        "steps": True,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--links", required=True, help="FIRST:LAST, of the set's 600")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--set", default="net2-train", choices=SETS_HERE)
    p.add_argument("--alone", action="store_true", help="only conditions without cross traffic")
    p.add_argument("--stock", default="bbr3", help="comma-separated stock controllers, or all")
    p.add_argument("--policy", action="append", default=[], help="NAME=PATH[@MODE]")
    a = p.parse_args()
    first, last = (int(x) for x in a.links.split(":"))
    ccs, aliases = controllers(a.stock, a.policy)
    spec = campaign(first, last, a.seed, ccs, set_=a.set, alone=a.alone)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "campaign.json").write_text(json.dumps(spec, indent=1) + "\n")
    (a.out / "aliases.json").write_text(json.dumps(aliases, indent=1) + "\n")
    alone = " without cross traffic" if a.alone else ""
    print(f"{last - first} {a.set} links, {len(ccs)} controllers, 2 conditions each{alone}")


if __name__ == "__main__":
    main()

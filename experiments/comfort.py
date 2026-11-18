"""Write a campaign on net2's own training links: examples/net1_broad.py's
net2-train set (600 links, a pool of 64 conditions, 30 s), whose step data
(exp-data-net2-steps-222) the offline net2s are trained on. Every link is
one the data had and every condition one of its pool; each link runs under
2 of them drawn with the campaign seed, so the pairing differs from the
data's. A one-off: the cost of the LCFs' meaning (experiments/lcf-cost.json)
is scored where both arms were trained, not on held-out links.

Usage: python experiments/comfort.py OUT_DIR --links 0:100 --seed 31
           --stock bbr3 --policy NAME=PATH ...

Writes OUT_DIR/campaign.json and OUT_DIR/aliases.json, as heldout.py does.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from heldout import controllers
from net1_broad import SETS, draw

SET = "net2-train"


def campaign(first: int, last: int, seed: int, ccs: list[dict[str, object]]) -> dict[str, object]:
    links, conditions = draw(SET)
    if not 0 <= first < last <= len(links):
        raise ValueError(f"links {first}:{last} are not within the set's {len(links)}")
    return {
        "name": f"comfort-{first}-{last}",
        "ccs": ccs,
        "sweep": {"kind": "paths", "paths": links[first:last]},
        "conditions": conditions,
        "conditions_per_point": 2,
        "duration_ms": SETS[SET][2] * 1000,
        "seed": seed,
        "steps": True,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--links", required=True, help="FIRST:LAST, of the set's 600")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stock", default="bbr3", help="comma-separated stock controllers, or all")
    p.add_argument("--policy", action="append", default=[], help="NAME=PATH[@MODE]")
    a = p.parse_args()
    first, last = (int(x) for x in a.links.split(":"))
    ccs, aliases = controllers(a.stock, a.policy)
    spec = campaign(first, last, a.seed, ccs)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "campaign.json").write_text(json.dumps(spec, indent=1) + "\n")
    (a.out / "aliases.json").write_text(json.dumps(aliases, indent=1) + "\n")
    print(f"{last - first} {SET} links, {len(ccs)} controllers, 2 conditions each")


if __name__ == "__main__":
    main()

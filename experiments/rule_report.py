"""Score the held-out campaigns of experiments/rule_eval.sh and, on
validation links, choose u for each group of rules.

Usage: python experiments/rule_report.py OUT [--select GROUP ...] [--best PREFIX ...]
           [--apart CAMPAIGN ...]

Scores every OUT/c-* campaign against BBRv3 (log utility per 10 ms step,
paired by link) and writes OUT/scores-CAMPAIGN.json and OUT/summary.md: one
row per controller, its difference from BBRv3 in each campaign and the mean
of those differences, which weights each campaign equally; campaigns named
in --apart are shown but left out of that mean. With --select,
policies named GROUP-uU are candidates and each group's u is the one with
the highest mean; a candidate with a failed episode is never chosen. The
choice goes to OUT/chosen.json ({GROUP: U}). With --best, the policy named
PREFIX-anything with the highest mean (never one with a failed episode) is
chosen for each prefix ({PREFIX: NAME} in OUT/chosen.json); with
--over-seeds, names ending -sN are one configuration over training seeds,
and the configuration with the highest mean over its seeds is chosen (never
one with a failed episode in any seed).
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from expliccit_env.train import score
from expliccit_env.train.spec import Net2Spec

REFERENCE = "bbr3"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--select", nargs="*", default=[])
    p.add_argument("--best", nargs="*", default=[], help="prefixes of policies to pick one of")
    p.add_argument("--apart", nargs="*", default=[], help="campaigns shown but not averaged")
    p.add_argument("--over-seeds", action="store_true", help="--best by the mean over -sN")
    a = p.parse_args()
    aliases: dict[str, str] = json.loads((a.out / "aliases.json").read_text())
    results: dict[str, dict[str, Any]] = {}
    for c in sorted(a.out.glob("c-*")):
        r = score.compare(
            score.episode_scores([c], Net2Spec(reward="log_utility")), REFERENCE, aliases
        )
        r["failed"] = {aliases.get(k, k): v for k, v in score.failures([c]).items()}
        (a.out / f"scores-{c.name[2:]}.json").write_text(json.dumps(r, indent=1) + "\n")
        results[c.name[2:]] = r
    names = sorted({n for r in results.values() for n in r["controllers"]} - {REFERENCE})
    failed = {n for r in results.values() for n, k in r["failed"].items() if k}

    def diff(n: str, camp: str) -> float | None:
        e = results[camp]["controllers"].get(n, {}).get("all", {}).get("minus_reference")
        return None if e is None else e["mean"]

    averaged = [c for c in results if c not in a.apart]

    def overall(n: str) -> float:
        ds = [diff(n, c) for c in averaged]
        return sum(d for d in ds if d is not None) / len(ds) if None not in ds else float("nan")

    # Campaigns kept apart come last.
    camps = averaged + [c for c in results if c in a.apart]
    apart = f" ({', '.join(a.apart)} not in the mean)" if a.apart else ""
    lines = [
        f"Log utility minus {REFERENCE}, paired by link, per campaign (all its conditions)"
        f" and the mean over campaigns{apart}. A * marks a controller with failed episodes.",
        "",
        "| controller | " + " | ".join(camps) + " | mean |",
        "|" + "---|" * (len(camps) + 2),
    ]
    for n in sorted(names, key=lambda n: -overall(n)):
        cells = []
        for c in camps:
            e = results[c]["controllers"].get(n, {}).get("all", {}).get("minus_reference")
            cells.append(
                ""
                if e is None
                else f"{e['mean']:+.3f} ({e['ci95'][0]:+.3f} to {e['ci95'][1]:+.3f})"
            )
        lines.append(
            f"| {n}{' *' if n in failed else ''} | " + " | ".join(cells) + f" | {overall(n):+.3f} |"
        )
    chosen: dict[str, float | str] = {}
    for g in a.select:
        cands = {
            float(m[1]): n for n in names if (m := re.fullmatch(rf"{re.escape(g)}-u(-?[0-9.]+)", n))
        }
        ok = {u: n for u, n in cands.items() if n not in failed and overall(n) == overall(n)}
        if ok:
            chosen[g] = max(ok, key=lambda u: overall(ok[u]))
    configs: dict[str, list[str]] = {}
    for n in names:
        configs.setdefault(re.sub(r"-s[0-9]+$", "", n) if a.over_seeds else n, []).append(n)
    mean = {
        k: sum(overall(n) for n in v) / len(v) for k, v in configs.items() if not failed & set(v)
    }
    for g in a.best:
        ok = [k for k, m in mean.items() if k.startswith(f"{g}-") and not math.isnan(m)]
        if ok:
            chosen[g] = max(ok, key=mean.__getitem__)
    if a.over_seeds:
        lines += ["", "| configuration | seeds | mean over seeds |", "|---|---|---|"]
        lines += [
            f"| {k} | {len(configs[k])} | {m:+.3f} |"
            for k, m in sorted(mean.items(), key=lambda km: -km[1])
            if len(configs[k]) > 1
        ]
    if a.select or a.best:
        (a.out / "chosen.json").write_text(json.dumps(chosen, indent=1) + "\n")
        lines += [
            "",
            "Chosen, by the mean over campaigns: "
            + ", ".join(
                f"{g} {u:g}" if isinstance(u, float) else f"{g} {u}" for g, u in chosen.items()
            )
            + ".",
        ]
    for c, r in results.items():
        lines += ["", f"## {c}", "", score.markdown(r)]
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

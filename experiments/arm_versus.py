"""One configuration minus another, each averaged over its training seeds,
paired by link.

Usage: python experiments/arm_versus.py OUT A:B ...

A and B name configurations: the policies named A-sN are A's seeds, and a
name without seeds (bbr3) is that controller. A link's score is the mean
log utility per 10 ms step of its episodes, averaged over the link's
conditions and then over the configuration's seeds; the difference is taken
per link and its mean shown with a 95% bootstrap interval over links, as
experiments/versus_rule.py does. Reads OUT/c-* and OUT/aliases.json; writes
OUT/versus.md.
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from net2_ppo_gate_report import _interval, load  # pyright: ignore[reportPrivateUsage]

type ByCondition = dict[str, dict[str, dict[str, float]]]


def link_scores(by_cond: ByCondition, ctl: str) -> dict[str, float]:
    """Each link's score, the mean over its conditions."""
    conds = by_cond[ctl].values()
    links = {k for c in conds for k in c}
    return {k: float(np.mean([c[k] for c in conds if k in c])) for k in links}


def config(by_cond: ByCondition, name: str) -> tuple[list[str], dict[str, float]]:
    """The configuration's policies and each link's mean over them."""
    seeds = sorted(n for n in by_cond if re.fullmatch(rf"{re.escape(name)}-s[0-9]+", n))
    seeds = seeds or ([name] if name in by_cond else [])
    if not seeds:
        raise SystemExit(f"no controller {name} or {name}-sN")
    per = [link_scores(by_cond, s) for s in seeds]
    links = set(per[0]).intersection(*per[1:])
    return seeds, {k: float(np.mean([p[k] for p in per])) for k in links}


def main(out: Path, pairs: list[str]) -> None:
    scores, failed = load(out)
    rng = np.random.default_rng(0)
    lines = [
        "A minus B, each averaged over its training seeds, paired by link: the mean over links",
        "with its 95% bootstrap interval. A * marks a configuration with a failed episode.",
        "",
        "| campaign | A (policies) | B (policies) | A minus B | links |",
        "|---|---|---|---|---|",
    ]
    for camp, by_cond in scores.items():
        for pair in pairs:
            a, b = pair.split(":")
            (sa, xa), (sb, xb) = config(by_cond, a), config(by_cond, b)
            shared = sorted(set(xa) & set(xb))
            iv = _interval([xa[k] - xb[k] for k in shared], rng)
            mark = {n: " *" if failed & set(s) else "" for n, s in ((a, sa), (b, sb))}
            lines.append(
                f"| {camp} | {a}{mark[a]} ({len(sa)}) | {b}{mark[b]} ({len(sb)})"
                f" | {iv} | {iv.links} |"
            )
    (out / "versus.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])

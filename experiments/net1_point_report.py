"""Compare single-value net1 runs (experiments/net1_train.py outputs) by arm.

Usage: python experiments/net1_point_report.py [--compare-name NAME] OUT_DIR RUN_DIR...

A run directory is named net1-point-ARM-sSEED. Per arm, view and LCF this
gives the R^2 of the belief's mean, as the mean over the arm's seeds with
its range, beside the comparison net1s each run scored on the same rows
(net1_train.py's --compare, called NAME in the table, and
--compare-expliccit). The arm with the
highest mean R^2 over the LCFs on the validation rows is chosen, the only
rows a choice may use. Writes OUT_DIR/summary.md and OUT_DIR/chosen.txt
(the chosen arm's seed-0 run).
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

NAMES = {"compare": "net1 v6 (64 bins, 3 members)", "expliccit": "expliCCit's released net1"}


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def cell(xs: list[float]) -> str:
    if len(xs) == 1:
        return f"{xs[0]:.2f}"
    return f"{mean(xs):.2f} ({min(xs):.2f} to {max(xs):.2f})"


def main(out: Path, runs: list[Path]) -> None:
    arms: dict[str, list[dict[str, Any]]] = {}
    for r in sorted(runs):
        m = re.fullmatch(r"net1-point-(.+)-s(\d+)", r.name)
        if m is None:
            raise SystemExit(f"{r.name}: not net1-point-ARM-sSEED")
        arms.setdefault(m[1], []).append(json.loads((r / "report.json").read_text()))
    first = next(iter(arms.values()))[0]
    lcfs = list(first["spec"]["lcfs"])

    def r2(reps: list[dict[str, Any]], view: str, model: str, lcf: str) -> list[float]:
        return [rep["views"][view][model][lcf]["r2"] for rep in reps]

    score = {a: mean([mean(r2(reps, "val", "net1", k)) for k in lcfs]) for a, reps in arms.items()}
    best = max(score, key=lambda a: score[a])
    lines = [
        "Single-value net1 (one value per LCF, squared error): R^2 of the value per LCF,",
        "the mean over seeds with its range. The other net1s are scored on the same rows",
        "by the mean of their belief.",
        "",
        "| arm | seeds | epochs | mean R^2 over the LCFs, validation rows |",
        "|---|---|---|---|",
    ]
    for a, reps in arms.items():
        ep = ", ".join(str(rep["log"]["epochs"]) for rep in reps)
        lines.append(f"| {a} | {len(reps)} | {ep} | {score[a]:.3f} |")
    lines += ["", f"Chosen on the validation rows: {best}.", ""]
    lines += ["| view | model | " + " | ".join(lcfs) + " |", "|" + "---|" * (len(lcfs) + 2)]
    for view in first["views"]:
        for a, reps in arms.items():
            cells = [cell(r2(reps, view, "net1", k)) for k in lcfs]
            lines.append(f"| {view} | point, {a} | " + " | ".join(cells) + " |")
        for model, name in NAMES.items():
            if model in first["views"][view]:
                # Every run scores the same comparison net1 on the same rows.
                cells = [cell(r2([first], view, model, k)) for k in lcfs]
                lines.append(f"| {view} | {name} | " + " | ".join(cells) + " |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "chosen.txt").write_text(f"net1-point-{best}-s0\n")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[0] == "--compare-name":
        NAMES["compare"], argv = argv[1], argv[2:]
    main(Path(argv[0]), [Path(p) for p in argv[1:]])

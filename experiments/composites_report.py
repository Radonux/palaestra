"""Does a composite LCF need its own head? Its head's R^2 beside the R^2 of
its formula applied to net1's estimates of its parts.

Usage: python experiments/composites_report.py OUT_DIR RUN_DIR...

Each run directory holds the report.json of experiments/net1_train.py for a
net1 whose heads include composites and their parts (fit.report_net1's
"derived"). Per view and composite this gives both R^2 values as the mean
over the runs with its range. Writes OUT_DIR/summary.md, and
OUT_DIR/composites.json with, for the validation rows only, the mean of
each. A composite whose head is no better than its formula on the
validation rows is redundant as a head: a consumer can compute it.
"""

import json
import sys
from pathlib import Path
from typing import Any


def cell(xs: list[float]) -> str:
    return f"{sum(xs) / len(xs):.2f} ({min(xs):.2f} to {max(xs):.2f})"


def main(out: Path, runs: list[Path]) -> None:
    reps: list[dict[str, Any]] = [json.loads((r / "report.json").read_text()) for r in sorted(runs)]
    val = reps[0]["views"]["val"]["net1"]
    names = [k for k, v in val.items() if "derived_r2" in v]
    if not names:
        raise SystemExit("no composite with all its parts among the heads")
    lines = [
        "R^2 of each composite LCF: its own head, then its formula on the estimates of its",
        f"parts; the mean over {len(reps)} seeds with its range.",
        "",
        "| view | from | " + " | ".join(names) + " |",
        "|" + "---|" * (len(names) + 2),
    ]
    for view in reps[0]["views"]:
        for key, label in (("r2", "head"), ("derived_r2", "formula on parts")):
            cells = [cell([rep["views"][view]["net1"][k][key] for rep in reps]) for k in names]
            lines.append(f"| {view} | {label} | " + " | ".join(cells) + " |")
    chosen = {
        k: {
            key: sum(rep["views"]["val"]["net1"][k][key] for rep in reps) / len(reps)
            for key in ("r2", "derived_r2")
        }
        for k in names
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "composites.json").write_text(json.dumps(chosen, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])

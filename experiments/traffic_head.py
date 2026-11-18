"""Train net1's traffic head (train/traffic.py) on a campaign's steps.

Reads every c-* campaign under OUT whose steps tables keep the
observations, the observed flow one controller throughout, on training
links; reports the head's balanced accuracy on links held out of its
training (3 folds by link, as experiments/competitor_info.py), then trains
it on every link and saves it as OUT/traffic_head.pt with its width.

Usage: python experiments/traffic_head.py OUT NET1.pt
Writes OUT/summary.md and OUT/traffic_head.pt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from competitor_info import FOLDS, held_out, load

from expliccit_env.train import fit
from expliccit_env.train.traffic import CLASSES, fit_head, save_head


def main(out: Path, net1_path: Path) -> None:
    net1, _ = fit.load_net1(net1_path)
    raw, _, y, link, _, _ = load(out, net1)
    m = held_out(raw, y, link)
    head = fit_head(raw, y)
    save_head(head, out / "traffic_head.pt")
    counts = {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}
    text = (
        f"net1's traffic head ({', '.join(CLASSES)}) on training links: "
        f"{len(set(link.tolist()))} links, steps {counts}.\n\n"
        f"On links held out of training ({FOLDS} folds by link): balanced accuracy "
        f"{m['balanced_accuracy']:.3f}, copies called stock {m['copies_called_stock']:.3f}, "
        f"stock called copies {m['stock_called_copies']:.3f}.\n\n"
        "The head trained on every link is traffic_head.pt (width and state).\n"
    )
    (out / "summary.md").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

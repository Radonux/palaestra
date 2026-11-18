"""Write a campaign of the broad net1 dataset (net1 v4): many links, each
under two conditions drawn from a broad set, three controllers per link.

Usage: python examples/net1_broad.py OUT_DIR --set SET --models DIR

SET is one of:
  train30, train60  the training data, 30 s and 60 s episodes (competition
                    needs time to settle), on separate links
  ood-competitors   competing flows of controllers no training condition
                    has (BBRv3, Westwood)
  ood-loss          random loss of 3% to 6% (training: at most 2%)
  ood-rtt           one-way delay of 61 to 90 ms (training: 5 to 60 ms)
  ood-buffer        buffers of 8 to 16 BDP (training: 0.25 to 8)
  net2-train        net2's online rollouts on net1 v4: training links and
                    conditions, on links of their own (its controllers are
                    ignored there)
  net2-train-xl     the same with 16 times the links, so that nearly every
                    rollout of a learned net2 runs on a link it has not seen
  wide30, wide120   net1 v6's added data: one-way delay of 5 to 150 ms,
                    buffers of 0.25 to 32 BDP, base capacity of 1 to 200
                    Mbit/s, and 30 s and 120 s episodes, so that slow
                    dynamics (a Reno or Cubic cycle on a large BDP, many
                    capacity steps, late arrivals and departures) play out
  ood-rtt-wide      one-way delay of 151 to 250 ms, otherwise as wide30
  ood-buffer-wide   buffers of 32 to 64 BDP, otherwise as wide30

The held-out sets use training conditions or training links except in the
one family they change. Training links: base capacity 4 to 150 Mbit/s,
constant or stepping 0.25x to 4x every 2 to 20 s. Training conditions: up to
three competing TCP flows (any stock controller but BBRv3 and Westwood),
arriving at the start or later and some leaving early; UDP at 5% to 50% of
the link's capacity; random loss of 0.01% to 2%, and on the ACK path.

Controllers of the observed flow, three per link: the 17 stock ones, the four
delay-threshold policies (examples/policies) and the fixed rule of release
model-net1v3 (DIR/rule-u0.25.onnx), alone and with exploration, so net1 sees
the states net2 visits. BBRv3, Westwood and threshold-0.3 are held out of
net1's training rows when it is trained, as for net1 v3.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from expliccit_env.spec import STOCK_NAMES, CampaignSpec

ROOT = Path(__file__).resolve().parent
HELD_COMPETITORS = ("bbr3", "westwood")
COMPETITORS = tuple(n for n in STOCK_NAMES if n not in HELD_COMPETITORS)
# Weights of the competitors: the Internet's common controllers more often.
WEIGHT = {"cubic": 6.0, "bbr": 4.0, "reno": 2.0}
SETS = {  # name: (links, conditions, duration s, campaign seed)
    "train30": (2400, 64, 30, 4001),
    "train60": (1200, 64, 60, 4002),
    "ood-competitors": (150, 16, 30, 4003),
    "ood-loss": (150, 16, 30, 4004),
    "ood-rtt": (150, 64, 30, 4005),
    "ood-buffer": (150, 64, 30, 4006),
    "net2-train": (600, 64, 30, 4011),
    "net2-train-xl": (9600, 64, 30, 4012),
    "wide30": (1200, 64, 30, 4021),
    "wide120": (600, 64, 120, 4022),
    "ood-rtt-wide": (150, 64, 30, 4023),
    "ood-buffer-wide": (150, 64, 30, 4024),
}
# name: (one-way delay ms, buffer in BDP, base capacity Mbit/s), where it is
# not the training ranges (5 to 60, 0.25 to 8, 4 to 150).
RANGES = {
    "ood-rtt": ((61, 90), (0.25, 8.0), (4.0, 150.0)),
    "ood-buffer": ((5, 60), (8.0, 16.0), (4.0, 150.0)),
    "wide30": ((5, 150), (0.25, 32.0), (1.0, 200.0)),
    "wide120": ((5, 150), (0.25, 32.0), (1.0, 200.0)),
    "ood-rtt-wide": ((151, 250), (0.25, 32.0), (1.0, 200.0)),
    "ood-buffer-wide": ((5, 150), (32.0, 64.0), (1.0, 200.0)),
}


def logu(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def link(
    rng: np.random.Generator,
    delay: tuple[int, int],
    buffer_bdp: tuple[float, float],
    capacity: tuple[float, float] = (4.0, 150.0),
) -> dict[str, object]:
    b = round(logu(rng, *capacity), 1)
    if rng.random() < 0.4:
        levels = [b]
    else:
        m = float(rng.choice([0.25, 0.5, 2.0, 4.0]))
        levels = [b, round(min(b * m, 300.0), 1)]
    one_way = int(rng.integers(delay[0], delay[1] + 1))
    bdp = b * 1e6 * 2 * one_way / 1e3 / (1500 * 8)
    buf = int(np.clip(bdp * logu(rng, *buffer_bdp), 10, 16000))
    level_ms = int(rng.choice([2000, 5000, 10000, 20000]))  # drawn last, as before
    return {
        "trace": {"kind": "periodic_levels", "levels_mbps": levels, "level_ms": level_ms},
        "delay": {"mahak_compat_mrtt_ms": one_way},
        "buffer": {"packets": buf},
    }


def window(rng: np.random.Generator, duration_ms: int) -> dict[str, int]:
    """Start at 0 or later, and leave early or stay to the end."""
    start = 0 if rng.random() < 0.35 else int(rng.uniform(1000, 0.6 * duration_ms))
    if rng.random() < 0.3:
        return {"start_ms": start, "stop_ms": int(rng.uniform(start + 5000, duration_ms))}
    return {"start_ms": start}


def condition(rng: np.random.Generator, duration_ms: int, family: str) -> dict[str, object]:
    if family == "ood-loss":
        return {"loss": {"downlink": round(rng.uniform(0.03, 0.06), 4)}}
    names = HELD_COMPETITORS if family == "ood-competitors" else COMPETITORS
    w = np.array([WEIGHT.get(n, 1.0) for n in names])
    cross: list[dict[str, object]] = []
    if family == "ood-competitors" or rng.random() < 0.55:
        k = 1 + int(rng.choice(3, p=[0.6, 0.3, 0.1]))
        for cc in rng.choice(names, size=k, p=w / w.sum()):
            cross.append({"kind": "tcp", "cc": str(cc), **window(rng, duration_ms)})
    if family == "ood-competitors":
        return {"cross": cross}
    if rng.random() < 0.2:
        cross.append(
            {
                "kind": "udp_share",
                "share": round(rng.uniform(0.05, 0.5), 3),
                **window(rng, duration_ms),
            }
        )
    loss = {}
    if rng.random() < 0.35:
        loss["downlink"] = round(logu(rng, 1e-4, 2e-2), 5)
    if rng.random() < 0.1:
        loss["uplink"] = round(logu(rng, 1e-4, 1e-2), 5)
    return {"loss": loss, "cross": cross}


def controllers(models: Path) -> list[dict[str, object]]:
    def agent(path: Path, **extra: object) -> dict[str, object]:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "kind": "agent",
            "mode": "hold",
            "policy": {"path": str(path), "sha256": sha},
            **extra,
        }

    ccs: list[dict[str, object]] = [{"kind": "stock", "name": n} for n in STOCK_NAMES]
    for theta in (0.9, 0.6, 0.3, 0.1):
        ccs.append(agent(ROOT / "policies" / f"threshold-{theta}.onnx", noise_sigma=0.01))
    rule = models / "rule-u0.25.onnx"
    ccs += [agent(rule), agent(rule, noise_sigma=0.5, noise_hold_steps=20)]
    return ccs


def draw(name: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """A set's links and its pool of conditions, as its campaign has them."""
    n_links, n_cond, seconds, seed = SETS[name]
    rng = np.random.default_rng(seed)
    delay, buffer_bdp, capacity = RANGES.get(name, ((5, 60), (0.25, 8.0), (4.0, 150.0)))
    family = name if name in ("ood-competitors", "ood-loss") else "train"
    links = [link(rng, delay, buffer_bdp, capacity) for _ in range(n_links)]
    return links, [condition(rng, seconds * 1000, family) for _ in range(n_cond)]


def campaign(name: str, models: Path) -> dict[str, object]:
    _, _, seconds, seed = SETS[name]
    links, conditions = draw(name)
    return {
        "name": f"net1-broad-{name}",
        "ccs": controllers(models),
        "ccs_per_point": 3,
        "sweep": {"kind": "paths", "paths": links},
        "conditions": conditions,
        "conditions_per_point": 2,
        "duration_ms": seconds * 1000,
        "seed": seed,
        "steps": True,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--set", choices=list(SETS), required=True)
    p.add_argument("--models", type=Path, required=True, help="release model-net1v3, unpacked")
    a = p.parse_args()
    spec = campaign(a.set, a.models.resolve())
    n = len(CampaignSpec.model_validate(spec).episodes())
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "campaign.json").write_text(json.dumps(spec, indent=1) + "\n")
    print(f"{a.set}: {n} episodes")


if __name__ == "__main__":
    main()

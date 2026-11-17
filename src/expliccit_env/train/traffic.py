"""The other traffic on the link, estimated from the measurements.

net1's seven LCFs leave out what tells copies of the flow from a stock TCP
competitor: per step, a classifier of the other traffic reaches 0.709
balanced accuracy on held-out links from the raw measurements and 0.472
from net1 v7's beliefs (exp-competitor-info-v7-299). A TrafficHead is
that classifier as a part of net1: it reads the measurements through
net1's input scaler and states the probability of each class. A policy
graph takes it in its front end (graph.Front ``traffic``); a gate3 head
can take its beliefs from it (ppo.Head), and a spread target can stop
probing among copies with it (graph.SpreadTarget ``copies_gated``).
"""

# torch's own annotations leave many signatures partly unknown (from_numpy,
# manual_seed); every other strict check stays on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

type F32 = npt.NDArray[np.float32]
CLASSES: Final = ("none", "copies", "stock")


class TrafficHead(nn.Module):
    """net1's scaled measurements [n, width] -> [n, 3] probabilities of
    CLASSES."""

    mu: torch.Tensor
    sd: torch.Tensor

    def __init__(self, width: int, hidden: int = 128) -> None:
        super().__init__()
        self.register_buffer("mu", torch.zeros(width))
        self.register_buffer("sd", torch.ones(width))
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, len(CLASSES))
        )

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mu) / self.sd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(x), dim=1)


def fit_head(x: F32, y: npt.NDArray[np.int64], *, epochs: int = 30, seed: int = 0) -> TrafficHead:
    """A head trained on rows x (net1's scaled measurements) and classes y,
    each class weighted by the inverse of its count."""
    torch.manual_seed(seed)
    head = TrafficHead(x.shape[1])
    head.mu = torch.from_numpy(x.mean(0).astype(np.float32))
    head.sd = torch.from_numpy((x.std(0) + 1e-6).astype(np.float32))
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    w = torch.from_numpy(np.where(counts > 0, 1 / np.maximum(counts, 1), 0).astype(np.float32))
    loss_fn = nn.CrossEntropyLoss(weight=w / w.sum())
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(xt.shape[0])
        for i in range(0, xt.shape[0], 1024):
            b = perm[i : i + 1024]
            opt.zero_grad()
            loss_fn(head.logits(xt[b]), yt[b]).backward()
            opt.step()
    return head.eval()


def save_head(head: TrafficHead, path: Path) -> None:
    torch.save({"width": head.mu.shape[0], "state": head.state_dict()}, path)


def load_head(path: Path) -> TrafficHead:
    """A head saved by save_head (experiments/traffic_head.py)."""
    saved = torch.load(path, weights_only=True)
    head = TrafficHead(int(saved["width"]))
    head.load_state_dict(saved["state"])
    return head.eval()

"""A policy graph to ONNX under contract sage-v1 (policy.py)."""

# torch's own annotations leave many signatures partly unknown (from_numpy,
# manual_seed, onnx.export); every other strict check stays on.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import onnx
import torch

from expliccit_env.policy import OBS_DIM, TRUTH_DIM
from expliccit_env.train.graph import PolicyGraph


class _ByName(torch.nn.Module):
    """The graph with its inputs passed by name, for the ones it uses."""

    def __init__(self, graph: PolicyGraph, names: list[str]) -> None:
        super().__init__()
        self.graph, self.names = graph, names

    def forward(self, *xs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        kw = dict(zip(self.names, xs, strict=True))
        if "state" not in kw:
            return self.graph(**kw)
        return self.graph.step(kw.pop("obs"), kw.pop("state"), **kw)


def to_onnx(graph: PolicyGraph) -> bytes:
    """Input "obs" [1, 78], plus "state" [1, S] when net2 has memory, "truth"
    [1, TRUTH_DIM] when the graph takes LCFs from the truth and "cwnd"
    [1, 1] when it sets a window target; output "a" [1, 1], then the next state [1, S]
    when net2 has memory.

    ponytail: TorchScript exporter (deprecated in torch 2.x, simple ops that
    tract runs); move to dynamo=True with onnxscript when it is removed."""
    graph.eval()
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = Path(tmp) / "policy.onnx"
        args = (torch.zeros(1, OBS_DIM),)
        names = ["obs"]
        if graph.state_dim:
            args, names = (*args, torch.zeros(1, graph.state_dim)), [*names, "state"]
        if graph.uses_truth:
            args, names = (*args, torch.zeros(1, TRUTH_DIM)), [*names, "truth"]
        if graph.target is not None:
            args, names = (*args, torch.full((1, 1), 10.0)), [*names, "cwnd"]
        # In eval mode: the exporter restores the module's mode afterwards,
        # and a new module starts in training mode, which would switch the
        # caller's graph (BatchNorm, dropout) to training.
        torch.onnx.export(
            _ByName(graph, names).eval(),
            args,
            path,
            input_names=names,
            output_names=["a", "next_state"] if graph.state_dim else ["a"],
            opset_version=17,
            dynamo=False,
        )
        raw = path.read_bytes()
    onnx.checker.check_model(onnx.load_from_string(raw))
    return raw

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
from expliccit_env.train.models import PolicyGraph


def to_onnx(graph: PolicyGraph) -> bytes:
    """Input "obs" [1, 78], plus "truth" [1, 7] when the graph takes LCFs from
    the truth; one output "a" [1, 1].

    ponytail: TorchScript exporter (deprecated in torch 2.x, simple ops that
    tract runs); move to dynamo=True with onnxscript when it is removed."""
    graph.eval()
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = Path(tmp) / "policy.onnx"
        args = (torch.zeros(1, OBS_DIM),)
        names = ["obs"]
        if graph.uses_truth:
            args, names = (*args, torch.zeros(1, TRUTH_DIM)), [*names, "truth"]
        torch.onnx.export(
            graph,
            args,
            path,
            input_names=names,
            output_names=["a"],
            opset_version=17,
            dynamo=False,
        )
        raw = path.read_bytes()
    onnx.checker.check_model(onnx.load_from_string(raw))
    return raw

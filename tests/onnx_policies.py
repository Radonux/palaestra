"""Tiny ONNX policies obeying the sage-v1 contract (policy.py), for tests."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import onnx
from onnx import TensorProto, helper, numpy_helper

from expliccit_env.policy import OBS_DIM


def _model(
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    init: dict[str, npt.NDArray[np.float64]],
) -> bytes:
    graph = helper.make_graph(
        nodes,
        "policy",
        inputs,
        outputs,
        initializer=[numpy_helper.from_array(v.astype(np.float32), k) for k, v in init.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def _obs() -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, OBS_DIM])


def _tensor(name: str, width: int) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, width])


def linear(weights: npt.NDArray[np.float64], bias: float) -> bytes:
    """a = obs @ weights + bias, with weights of shape [OBS_DIM]."""
    return _model(
        [
            helper.make_node("MatMul", ["obs", "W"], ["h"]),
            helper.make_node("Add", ["h", "b"], ["a"]),
        ],
        [_obs()],
        [_tensor("a", 1)],
        {"W": weights.reshape(OBS_DIM, 1), "b": np.array([[bias]])},
    )


def constant(a: float) -> bytes:
    return linear(np.zeros(OBS_DIM), a)


def counter(slope: float) -> bytes:
    """Recurrent: state counts steps from 0, and a = slope * state."""
    return _model(
        [
            helper.make_node("Mul", ["state", "k"], ["a"]),
            helper.make_node("Add", ["state", "one"], ["next"]),
        ],
        [_obs(), _tensor("state", 1)],
        [_tensor("a", 1), _tensor("next", 1)],
        {"k": np.array([[slope]]), "one": np.array([[1.0]])},
    )

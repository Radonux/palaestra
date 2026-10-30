"""Tiny ONNX policies obeying the tcpinfo-v1 contract, for tests."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import onnx
from onnx import TensorProto, helper, numpy_helper

from expliccit_env.policy import ACTIONS, FEATURES


def linear(weights: npt.NDArray[np.float64], bias: npt.NDArray[np.float64]) -> bytes:
    """action = obs @ weights + bias."""
    obs = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, len(FEATURES)])
    act = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, len(ACTIONS)])
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["obs", "W"], ["h"]),
            helper.make_node("Add", ["h", "b"], ["action"]),
        ],
        "policy",
        [obs],
        [act],
        initializer=[
            numpy_helper.from_array(weights.astype(np.float32), "W"),
            numpy_helper.from_array(bias.astype(np.float32), "b"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def constant(cwnd: float, pacing_mbps: float = 0.0) -> bytes:
    return linear(np.zeros((len(FEATURES), 2)), np.array([cwnd, pacing_mbps]))

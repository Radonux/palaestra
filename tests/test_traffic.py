"""net1's traffic head (train/traffic.py) and the policy parts that read
it: the front end (graph.Front) and a gate3 head (actors.Head)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from expliccit_env.train import actors, fit, ppo  # noqa: E402
from expliccit_env.train.spec import Gate3, U  # noqa: E402
from expliccit_env.train.traffic import (  # noqa: E402
    CLASSES,
    TrafficHead,
    fit_head,
    load_head,
    save_head,
)

from .test_e2e import _onnx_matches  # noqa: E402
from .test_ppo import _bdp_net1  # noqa: E402


def test_the_head_learns_separable_classes() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, len(CLASSES), 3000)
    x = (rng.normal(size=(3000, 5)) + 3 * np.eye(5)[y]).astype(np.float32)
    head = fit_head(x, y.astype(np.int64), epochs=5)
    with torch.no_grad():
        p = head(torch.from_numpy(x))
    assert torch.allclose(p.sum(1), torch.ones(3000), atol=1e-5)
    assert (p.argmax(1).numpy() == y).mean() > 0.9


def test_a_gate3_head_takes_its_beliefs_from_the_traffic_head() -> None:
    _, net1 = _bdp_net1()
    spec = ppo.PPOSpec(
        paths_from=Path("x"), net1_from=Path("x"), algo="dagger", student=Gate3(), u_rule=0.25
    )
    plain = actors.Head(net1, spec)
    z = torch.from_numpy(net1.beliefs(np.zeros((1, 78), np.float32)))
    head = torch.tensor([[0.2, 0.7, 0.1]])  # none, copies, stock
    a = torch.tensor([[0.5, -1.0]])
    # q = 2 P(copies) - 1 replaces the student's; p stays the student's.
    want = plain(torch.tensor([[0.5, 0.4]]), z, None)
    assert torch.allclose(actors.Head(net1, spec, traffic_q=True)(a, z, head), want)
    # With p from the head too, p = 2 P(stock) - 1: no learned output is left.
    want = plain(torch.tensor([[-0.8, 0.4]]), z, None)
    assert torch.allclose(actors.Head(net1, spec, traffic_q=True, traffic_p=True)(a, z, head), want)
    with pytest.raises(ValueError, match="gate3"):
        actors.Head(net1, spec.model_copy(update={"student": U()}), traffic_q=True)


def test_the_front_end_states_the_traffic_beside_the_beliefs() -> None:
    _, net1 = _bdp_net1()
    head = TrafficHead(net1.scaler.cols.numel())
    obs = torch.randn(3, 78)
    z, x, traffic = fit.front(net1, traffic=head)(obs)
    assert traffic is not None and traffic.shape == (3, len(CLASSES))
    with torch.no_grad():
        assert torch.allclose(traffic, head(net1.scaler(obs)))
        assert torch.allclose(z, net1.net(net1.scaler(obs))) and torch.equal(x, z)


def test_a_saved_head_loads_unchanged(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 4)).astype(np.float32)
    head = fit_head(x, rng.integers(0, len(CLASSES), 200).astype(np.int64), epochs=1)
    save_head(head, tmp_path / "head.pt")
    with torch.no_grad():
        assert torch.equal(
            load_head(tmp_path / "head.pt")(torch.from_numpy(x)), head(torch.from_numpy(x))
        )


def test_a_spread_head_lowers_u_by_the_spread_where_no_stock_flow_is_believed() -> None:
    _, net1 = _bdp_net1()
    spec = ppo.PPOSpec(
        paths_from=Path("x"), net1_from=Path("x"), algo="dagger", student=Gate3(), u_rule=0.25
    )
    plain = actors.Head(net1, spec)
    z = torch.from_numpy(
        net1.beliefs(np.random.default_rng(0).normal(size=(4, 78)).astype(np.float32))
    )
    sd = actors.SpreadHead(plain, net1, spec, 1.0, 0.0).spread.spread(z)
    for p, scale in ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5)):
        a = torch.tensor([[2 * p - 1, 0.0]]).repeat(4, 1)
        got = actors.SpreadHead(plain, net1, spec, -0.8, 0.1)(a, z, None)
        want = plain(a, z, None) - 0.8 * torch.relu(sd - 0.1) * scale
        assert torch.allclose(got, want, atol=1e-6)
    for q in (0.0, 0.5, 1.0):  # gated by q, the belief in copies, instead
        a = torch.tensor([[1.0, 2 * q - 1]]).repeat(4, 1)
        got = actors.SpreadHead(plain, net1, spec, -0.8, 0.1, gate="copies")(a, z, None)
        want = plain(a, z, None) - 0.8 * torch.relu(sd - 0.1) * q
        assert torch.allclose(got, want, atol=1e-6)
    with pytest.raises(ValueError, match="gate3"):
        actors.SpreadHead(plain, net1, spec.model_copy(update={"student": U()}), -0.8, 0.1)


def test_a_net2_on_net1_and_its_traffic_head_reads_both_in_training_and_in_the_graph(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    _, net1 = _bdp_net1()
    path = tmp_path / "head.pt"
    save_head(TrafficHead(net1.scaler.cols.numel()), path)
    plain = ppo.PPOSpec(
        paths_from=Path("x"),
        net1_from=Path("x"),
        algo="dagger",
        student=Gate3(),
        u_rule=0.25,
        view="means",
    )
    spec = plain.model_validate(
        plain.model_dump() | {"inputs": "net1+traffic", "traffic_head": path}
    )
    with pytest.raises(ValueError, match="traffic_head"):  # the input without its head
        plain.model_validate(plain.model_dump() | {"inputs": "net1+traffic"})
    with pytest.raises(ValueError, match="traffic_head"):  # a head nothing reads
        plain.model_validate(plain.model_dump() | {"traffic_head": path})
    obs = torch.randn(5, 78)
    _, x, traffic = actors.front(net1, spec)(obs)
    _, x0, _ = actors.front(net1, plain)(obs)
    assert traffic is not None
    with torch.no_grad():  # net1's view first, then the head's probabilities
        assert torch.allclose(x, torch.cat([x0, traffic], dim=1))
    assert actors.width(net1, spec) == actors.width(net1, plain) + len(CLASSES)
    with pytest.raises(ValueError, match="traffic_head"):  # one head, the spec's
        actors.front(net1, spec, traffic=load_head(path))
    actor = actors.new_actor(actors.width(net1, spec), spec)
    graph = ppo.policy_graph(net1, actor, spec)
    rng = np.random.default_rng(0)
    _onnx_matches(graph, rng.normal(size=(40, 78)).astype(np.float32) + 20)

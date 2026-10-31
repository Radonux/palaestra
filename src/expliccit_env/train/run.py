"""Training runs and imports, as the CLI calls them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from expliccit_env.train import data, fit
from expliccit_env.train.export import to_onnx
from expliccit_env.train.spec import TrainSpec


def _write(out: Path, report: dict[str, Any]) -> None:
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))


def train(spec: TrainSpec, out: Path) -> dict[str, Any]:
    """net1, then net2 on it, then the policy graph; writes OUT/{net1.pt,
    net2.pt, policy.onnx, report.json, train_spec.json}."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_spec.json").write_text(spec.model_dump_json(indent=2))
    steps = data.load(
        spec.data,
        seed=spec.seed,
        val=spec.val_fraction,
        test=spec.test_fraction,
        controllers=spec.controllers,
    )
    if spec.net1_from is None:
        net1, net1_log = fit.fit_net1(spec.net1, steps, spec.seed)
        net1_spec = spec.net1
    else:
        net1, net1_spec = fit.load_net1(spec.net1_from)
        net1_log = {"loaded_from": str(spec.net1_from)}
    fit.save_net1(net1, net1_spec, out / "net1.pt")
    actor, net2_log = fit.fit_net2(spec.net2, steps, net1, spec.seed)
    torch.save(actor.state_dict(), out / "net2.pt")
    report: dict[str, Any] = {
        "rows": {s: int(steps.rows(s).sum()) for s in ("train", "val", "test")},
        "episodes": len(steps.episode_ids),
        "net1": {"training": net1_log, "test": fit.report_net1(net1, steps, "test")},
        "net2": net2_log,
    }
    # A net2 on the true LCFs runs on the guest's live truth (an oracle).
    truth = frozenset(net1.lcfs) if spec.net2.inputs == "true_lcfs" else frozenset[str]()
    graph = fit.policy_graph(
        net1, actor, spec.net2.action_max, spec.net2.action_decimals, truth=truth
    )
    (out / "policy.onnx").write_bytes(to_onnx(graph))
    _write(out, report)
    return report


def import_expliccit(
    models: Path,
    out: Path,
    data_dirs: list[Path],
    *,
    as_deployed: bool,
    truth: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """expliCCit's released net1 and actor as a sage-v1 policy (actions
    in [-1, 1], as its tanh gives), and net1's error on our data. The LCFs
    in ``truth`` come from the guest's live truth instead of net1, as in
    expliCCit's own repair experiment."""
    out.mkdir(parents=True, exist_ok=True)
    net1, actor = fit.load_expliccit(models)
    graph = fit.policy_graph(net1, actor, 1.0, truth=truth)
    graph.as_deployed = as_deployed
    (out / "policy.onnx").write_bytes(to_onnx(graph))
    report: dict[str, Any] = {"as_deployed": as_deployed, "from_truth": sorted(truth)}
    if data_dirs:
        steps = data.load(data_dirs, seed=0, val=0.0, test=0.0)
        report["net1_on_data"] = fit.report_net1(net1, steps)
        report["rows"] = int(steps.obs.shape[0])
    _write(out, report)
    return report

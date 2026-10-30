"""Spec parsing laws: invalid combinations are unrepresentable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from expliccit_env.spec import BaseRtt, CampaignSpec, EpisodeSpec, MahakMrtt

EXAMPLES = Path(__file__).parents[1] / "examples"


def episode(delay: object, **extra: object) -> str:
    return json.dumps(
        {
            "path": {
                "trace": {"kind": "mahak_step", "bw_mbps": 10, "step": 1.0},
                "delay": delay,
                "buffer": {"packets": 100},
            },
            "cc": {"kind": "stock", "name": "cubic"},
            **extra,
        }
    )


def test_delay_is_exactly_one_kind() -> None:
    assert EpisodeSpec.parse(episode({"mahak_compat_mrtt_ms": 20})).path.delay == MahakMrtt(
        mahak_compat_mrtt_ms=20
    )
    assert EpisodeSpec.parse(episode({"base_rtt_ms": 40})).path.delay == BaseRtt(base_rtt_ms=40)
    with pytest.raises(ValidationError):
        EpisodeSpec.parse(episode({"mahak_compat_mrtt_ms": 20, "base_rtt_ms": 40}))
    with pytest.raises(ValidationError):
        EpisodeSpec.parse(episode({}))


def test_mahak_mrtt_is_one_way() -> None:
    a = EpisodeSpec.parse(episode({"mahak_compat_mrtt_ms": 20}))
    b = EpisodeSpec.parse(episode({"base_rtt_ms": 40}))
    assert a.path.delay.one_way_ns == b.path.delay.one_way_ns == 20_000_000


def test_rejects_unknown_fields_and_bad_values() -> None:
    with pytest.raises(ValidationError):
        EpisodeSpec.parse(episode({"base_rtt_ms": 40}, surprise=1))
    with pytest.raises(ValidationError):
        EpisodeSpec.parse(episode({"base_rtt_ms": 0}))
    bad_buffer = json.loads(episode({"base_rtt_ms": 40}))
    bad_buffer["path"]["buffer"] = {"packets": 0, "bytes": 0}
    with pytest.raises(ValidationError):
        EpisodeSpec.parse(json.dumps(bad_buffer))


def test_episode_id_is_a_content_hash() -> None:
    a = EpisodeSpec.parse(episode({"base_rtt_ms": 40}, seed=3))
    assert a.episode_id() == EpisodeSpec.parse(a.canonical_json()).episode_id()
    assert a.episode_id() != EpisodeSpec.parse(episode({"base_rtt_ms": 40}, seed=4)).episode_id()


def test_campaign_expansion_is_deterministic() -> None:
    spec = CampaignSpec.parse((EXAMPLES / "campaign-icnp25-sample.json").read_text())
    a, b = spec.episodes(), spec.episodes()
    assert len(a) == 2000
    assert [e.canonical_json() for e in a] == [e.canonical_json() for e in b]
    assert len({e.episode_id() for e in a}) == len(a)
    smoke = CampaignSpec.parse((EXAMPLES / "campaign-smoke.json").read_text()).episodes()
    assert len(smoke) == 8


def test_examples_parse() -> None:
    EpisodeSpec.parse((EXAMPLES / "episode-mahak.json").read_text()).guest_plan()

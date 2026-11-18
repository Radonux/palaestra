"""experiments/heldout.py: the families where information has value."""

import importlib.util
from pathlib import Path

import pytest

from expliccit_env.spec import CampaignSpec

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "heldout.py"
_spec = importlib.util.spec_from_file_location("heldout", _PATH)
assert _spec is not None and _spec.loader is not None
heldout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(heldout)


@pytest.mark.parametrize("duration_ms", [20000, 30000, 60000])
def test_info_windows_lie_inside_the_episode(duration_ms: int) -> None:
    conditions = heldout.info(duration_ms)
    assert len(conditions) == 4
    for c in conditions:
        for f in c["cross"]:
            assert 0 <= f["start_ms"] < f["stop_ms"] <= duration_ms
    on_off_udp, on_off_tcp = conditions[2]["cross"], conditions[3]["cross"]
    assert all(f["stop_ms"] - f["start_ms"] == 2000 for f in on_off_udp)
    assert all(f["stop_ms"] - f["start_ms"] == 3000 for f in on_off_tcp)
    assert len(on_off_udp) == (duration_ms - 4000) // 4000 + 1  # from 2 s, every 4 s


def test_rise_alternates_half_and_twice_the_first_level() -> None:
    p = {"trace": {"kind": "periodic_levels", "levels_mbps": [24.0, 48.0], "level_ms": 7000}}
    (out,) = heldout.rise([p])
    assert out["trace"] == {
        "kind": "periodic_levels",
        "levels_mbps": [12.0, 48.0],
        "level_ms": 2000,
    }
    assert p["trace"]["level_ms"] == 7000  # the source link is not changed


_VPATH = Path(__file__).resolve().parents[1] / "experiments" / "versus_rule.py"
_vspec = importlib.util.spec_from_file_location("versus_rule", _VPATH)
assert _vspec is not None and _vspec.loader is not None
versus_rule = importlib.util.module_from_spec(_vspec)
_vspec.loader.exec_module(versus_rule)


def test_conditions_group_without_their_udp_rates() -> None:
    a = versus_rule.group("UDP 12.5 Mbit/s from 2 s, UDP 12.5 Mbit/s from 6 s")
    b = versus_rule.group("UDP 3 Mbit/s from 2 s, UDP 3 Mbit/s from 6 s")
    assert a == b == "UDP from 2 s, UDP from 6 s"
    assert versus_rule.group("cubic from 0 s, loss 1%") == "cubic from 0 s, loss 1%"


def test_versus_rule_pairs_by_link_and_groups_udp_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # campaign -> controller -> condition -> link -> score
    scores = {
        "info-30": {
            "rule-v6": {
                "UDP 4 Mbit/s from 2 s": {"a": 1.0},
                "UDP 8 Mbit/s from 2 s": {"b": 2.0},
                "cubic from 0 s": {"a": 0.0, "b": 0.0},
            },
            "spread1": {
                "UDP 4 Mbit/s from 2 s": {"a": 1.5},
                "UDP 8 Mbit/s from 2 s": {"b": 2.5},
                "cubic from 0 s": {"a": 0.1, "b": 0.1},
            },
        }
    }
    monkeypatch.setattr(versus_rule, "load", lambda out: (scores, set()))
    versus_rule.main(tmp_path, "rule-v6")
    text = (tmp_path / "versus-rule-v6.md").read_text()
    # Per link, the mean over its conditions: (+0.5 + 0.1) / 2 on both links.
    assert "| info-30 | all | +0.300 (+0.300 to +0.300) |" in text
    # The two UDP rates are one condition, paired over both links.
    assert "| info-30 | UDP from 2 s | +0.500 (+0.500 to +0.500) |" in text
    assert "| info-30 | cubic from 0 s | +0.100 (+0.100 to +0.100) |" in text


def test_a_trace_slice_starts_at_zero_and_keeps_the_window(tmp_path: Path) -> None:
    src = tmp_path / "t.down"
    src.write_text("".join(f"{t}\n" for t in (5, 10, 10, 20, 35, 40, 60)))
    spec = heldout.trace_slice(src, 10, 30, tmp_path / "s.down")
    assert (tmp_path / "s.down").read_text().split() == ["10", "25", "30"]
    assert spec["kind"] == "mahimahi_file" and len(spec["sha256"]) == 64


def test_cellular_links_keep_delay_and_buffer_in_bdps(tmp_path: Path) -> None:
    p = {
        "trace": {"kind": "periodic_levels", "levels_mbps": [24.0], "level_ms": 1000},
        "delay": {"mahak_compat_mrtt_ms": 40},
        "buffer": {"packets": 320},  # 2 BDPs of 24 Mbit/s at 80 ms (160 packets each)
    }
    links = heldout.cellular([p] * 9, 30000, tmp_path)
    names = [Path(x["trace"]["path"]).name for x in links]
    assert names[0] == "ATT-LTE-driving-0.down" and names[8] == "ATT-LTE-driving-37000.down"
    assert all(x["delay"] == p["delay"] for x in links)
    assert all(Path(x["uplink_trace"]["path"]).name.endswith(".up") for x in links)
    first = heldout.CELLULAR_DIR / "ATT-LTE-driving.down"
    bdp = heldout.mean_mbps(first) * 1e6 * 0.08 / 12000
    assert links[0]["buffer"]["packets"] == max(10, round(2 * bdp))


_CPATH = Path(__file__).resolve().parents[1] / "experiments" / "comfort.py"
_cspec = importlib.util.spec_from_file_location("comfort", _CPATH)
assert _cspec is not None and _cspec.loader is not None
comfort = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(comfort)


def test_comfort_runs_net2s_training_links_under_its_conditions() -> None:
    links, conditions = comfort.draw("net2-train")
    spec = comfort.campaign(100, 110, 32, [{"kind": "stock", "name": "bbr3"}])
    assert spec["sweep"]["paths"] == links[100:110]
    assert spec["conditions"] == conditions and len(conditions) == 64
    episodes = CampaignSpec.model_validate(spec).episodes()
    assert len(episodes) == 10 * 2 and all(e.duration_ms == 30000 for e in episodes)
    with pytest.raises(ValueError, match="within"):
        comfort.campaign(590, 601, 32, [])


def test_comfort_can_draw_links_and_conditions_the_data_never_had() -> None:
    train_links, train_conditions = comfort.draw("net2-train")
    spec = comfort.campaign(0, 10, 33, [{"kind": "stock", "name": "bbr3"}], set_="net2-train-xl")
    assert spec["name"] == "net2-train-xl-0-10" and len(spec["conditions"]) == 64
    assert not any(p in train_links for p in spec["sweep"]["paths"])
    # A new draw of conditions; the simplest (the flow alone) recur by chance.
    assert sum(c in train_conditions for c in spec["conditions"]) < 32


def test_comfort_alone_keeps_only_the_conditions_without_cross_traffic() -> None:
    _, conditions = comfort.draw("net2-train")
    spec = comfort.campaign(100, 110, 34, [{"kind": "stock", "name": "bbr3"}], alone=True)
    assert spec["name"] == "comfort-100-110-alone"
    assert spec["conditions"] == [c for c in conditions if not c["cross"]]
    assert len(spec["conditions"]) == 27
    episodes = CampaignSpec.model_validate(spec).episodes()
    assert len(episodes) == 10 * 2 and not any(e.cross for e in episodes)

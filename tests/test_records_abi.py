"""The numpy dtypes in records.py must lay out exactly like the C structs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from expliccit_env import records

HEADER = Path(__file__).parents[1] / "kernel" / "modules" / "expliccit_rec.h"
STRUCTS = {
    "expl_rec_meta": records.META,
    "expl_rec_link": records.LINK,
    "expl_rec_opp": records.OPP,
    "expl_rec_link_stats": records.STATS,
    "expl_rec_tcp_obs": records.OBS,
}


def _c_layout(tmp_path: Path) -> dict[str, tuple[int, dict[str, int]]]:
    lines = [
        "#include <stdio.h>",
        "#include <stddef.h>",
        f'#include "{HEADER}"',
        "int main(void) {",
    ]
    for struct, dtype in STRUCTS.items():
        lines.append(f'printf("{struct} size %zu\\n", sizeof(struct {struct}));')
        for name in dtype.names or ():
            member = f"hdr.{name}" if name in ("type", "len", "seq", "t_ns") else name
            lines.append(f'printf("{struct} {name} %zu\\n", offsetof(struct {struct}, {member}));')
    lines += ["return 0;", "}"]
    src = tmp_path / "layout.c"
    src.write_text("\n".join(lines))
    exe = tmp_path / "layout"
    subprocess.run(["cc", "-Wall", "-Werror", "-o", str(exe), str(src)], check=True)
    out: dict[str, tuple[int, dict[str, int]]] = {}
    for line in subprocess.run(
        [str(exe)], check=True, capture_output=True, text=True
    ).stdout.splitlines():
        struct, name, value = line.split()
        size, offsets = out.get(struct, (0, {}))
        if name == "size":
            size = int(value)
        else:
            offsets[name] = int(value)
        out[struct] = (size, offsets)
    return out


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")
def test_dtypes_match_c_structs(tmp_path: Path) -> None:
    layout = _c_layout(tmp_path)
    for struct, dtype in STRUCTS.items():
        size, offsets = layout[struct]
        assert dtype.itemsize == size, struct
        assert {n: dtype.fields[n][1] for n in dtype.names or ()} == offsets, struct
        assert size % 8 == 0, struct


def test_load_rejects_sequence_gap(tmp_path: Path) -> None:
    meta = np.zeros(1, records.META)
    meta["type"], meta["len"], meta["magic"], meta["abi"], meta["seq"] = (
        1,
        32,
        records.MAGIC,
        records.ABI,
        5,
    )
    obs = np.zeros(2, records.OBS)
    obs["type"], obs["len"], obs["seq"] = 7, 160, [6, 8]
    for stem, _, _ in records.FILES:
        (tmp_path / f"{stem}.bin").write_bytes(b"")
    (tmp_path / "meta.bin").write_bytes(meta.tobytes())
    (tmp_path / "tcp_obs.bin").write_bytes(obs.tobytes())
    with pytest.raises(records.RecordError, match="contiguous"):
        records.load_records(tmp_path)
    obs["seq"] = [6, 7]
    (tmp_path / "tcp_obs.bin").write_bytes(obs.tobytes())
    assert records.load_records(tmp_path).tcp_obs.size == 2


def test_load_rejects_truncated_file(tmp_path: Path) -> None:
    for stem, _, _ in records.FILES:
        (tmp_path / f"{stem}.bin").write_bytes(b"")
    (tmp_path / "link_opp.bin").write_bytes(b"\0" * 41)
    with pytest.raises(records.RecordError, match="multiple"):
        records.load_records(tmp_path)


def test_step_record_matches_the_guest() -> None:
    """steps.STEP must lay out like the record guest/src/step.rs writes."""
    import re  # noqa: PLC0415

    from expliccit_env import policy, steps  # noqa: PLC0415

    src = (Path(__file__).parents[1] / "guest" / "src" / "step.rs").read_text()

    def const(name: str) -> int:
        m = re.search(rf"pub const {name}: usize = ([0-9_]+);", src)
        assert m, name
        return int(m.group(1).replace("_", ""))

    assert const("OBS") == policy.OBS_DIM
    assert const("INFO_LEN") == steps.INFO_LEN
    assert steps.STEP.itemsize == 8 + steps.INFO_LEN + 8 * policy.OBS_DIM + 8 + 4 + 4
    assert [steps.STEP.fields[f][1] for f in ("t_us", "info", "obs", "applied_action")] == [
        0,
        8,
        8 + steps.INFO_LEN,
        8 + steps.INFO_LEN + 8 * policy.OBS_DIM,
    ]

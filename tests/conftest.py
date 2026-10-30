"""Shared fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from expliccit_env.uml import UmlBuild


@pytest.fixture(scope="session")
def build() -> UmlBuild:
    """A built UML tree (scripts/build-uml.sh, then build-guest.sh)."""
    path = os.environ.get("EXPLICCIT_UML_DIR")
    if not path:
        pytest.skip("set EXPLICCIT_UML_DIR to a built UML tree")
    return UmlBuild.at(Path(path))

from __future__ import annotations

import pathlib

import pytest

tomllib = pytest.importorskip("tomllib")

from scripts.build_core_package import RUN_DEPENDENCIES, core_pyproject  # noqa: E402

_PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_core_package_is_mteb_without_the_dependencies_to_run_models() -> None:
    """`mteb-core` is `mteb` without torch, transformers and sentence-transformers; everything else is kept."""
    pyproject = _PYPROJECT.read_text(encoding="utf-8")
    mteb = tomllib.loads(pyproject)["project"]
    core = tomllib.loads(core_pyproject(pyproject))["project"]
    left_out = [d for d in mteb["dependencies"] if d not in core["dependencies"]]

    assert core["name"] == "mteb-core"
    assert core["version"] == mteb["version"]
    assert len(left_out) == len(RUN_DEPENDENCIES)
    assert core["optional-dependencies"] == mteb["optional-dependencies"]

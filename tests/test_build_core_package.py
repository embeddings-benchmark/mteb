from __future__ import annotations

import pathlib

import pytest

tomllib = pytest.importorskip("tomllib")

from scripts.build_core_package import RUN_DEPENDENCIES, core_pyproject  # noqa: E402

_PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_core_package_moves_run_dependencies_into_an_extra() -> None:
    """`mteb-core` is `mteb` with torch, transformers and sentence-transformers moved into its `run` extra."""
    pyproject = _PYPROJECT.read_text(encoding="utf-8")
    mteb = tomllib.loads(pyproject)["project"]
    core = tomllib.loads(core_pyproject(pyproject))["project"]
    run = [d for d in mteb["dependencies"] if d not in core["dependencies"]]

    assert core["name"] == "mteb-core"
    assert core["version"] == mteb["version"]
    assert len(run) == len(RUN_DEPENDENCIES)
    assert core["optional-dependencies"] == {
        "run": run,
        **mteb["optional-dependencies"],
    }

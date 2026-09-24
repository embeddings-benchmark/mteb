"""Build `mteb-core` into `dist/`, next to the `mteb` distributions.

`mteb-core` ships the same code as `mteb`, but without torch, transformers and sentence-transformers, which move
into its `run` extra. It is for working with tasks, benchmarks, model metadata and results without installing
torch. Install either `mteb` or `mteb-core`, not both, as they contain the same files.

It is generated from the root `pyproject.toml` at release time, so adding a dependency or an extra only ever
means editing that file.

Usage:
    python scripts/build_core_package.py [--outdir dist]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent

# only needed to load and run models
RUN_DEPENDENCIES = {"torch", "transformers", "sentence-transformers"}


def core_pyproject(pyproject: str) -> str:
    """Return the `pyproject.toml` of `mteb-core` for the given `pyproject.toml` of `mteb`."""
    project = tomllib.loads(pyproject)["project"]
    if project["name"] != "mteb":
        raise ValueError(
            f"expected the root project to be 'mteb', got {project['name']!r}"
        )

    run = [
        dependency
        for dependency in project["dependencies"]
        if canonicalize_name(Requirement(dependency).name) in RUN_DEPENDENCIES
    ]
    found = {canonicalize_name(Requirement(dependency).name) for dependency in run}
    if found != RUN_DEPENDENCIES:
        raise ValueError(
            f"dependencies {RUN_DEPENDENCIES - found} are missing from pyproject.toml"
        )

    core = pyproject.replace('\nname = "mteb"\n', '\nname = "mteb-core"\n', 1)
    for dependency in run:
        line = f'    "{dependency}",\n'
        if line not in core:
            raise ValueError(
                f"expected {line.strip()!r} on its own line in dependencies"
            )
        core = core.replace(line, "", 1)
    run_extra = (
        "run = [\n" + "".join(f'    "{dependency}",\n' for dependency in run) + "]\n"
    )
    return core.replace(
        "[project.optional-dependencies]\n",
        "[project.optional-dependencies]\n" + run_extra,
        1,
    )


def build(source: Path, outdir: Path) -> None:
    """Build the sdist and wheel of `source` into `outdir`, in an isolated build environment."""
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(outdir.resolve()), str(source)],
        check=True,
    )


def build_core_package(outdir: Path) -> None:
    """Build `mteb-core` from the working tree into `outdir`."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        (src / "pyproject.toml").write_text(core_pyproject(pyproject), encoding="utf-8")
        for name in ("README.md", "LICENSE"):
            shutil.copy(REPO_ROOT / name, src / name)
        shutil.copytree(
            REPO_ROOT / "mteb",
            src / "mteb",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        build(src, outdir)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "dist")
    build_core_package(parser.parse_args().outdir)


if __name__ == "__main__":
    main()

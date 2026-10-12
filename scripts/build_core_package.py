"""Build `mteb-core` into `dist/`, next to the `mteb` distributions.

`mteb-core` ships the same code as `mteb`, but without torch, transformers and sentence-transformers, so it is
a minimal install for working with tasks, benchmarks, model metadata and results. Loading and running models
needs `mteb`. Install either one, not both, as they contain the same files.

It is generated from the root `pyproject.toml` at release time, so adding a dependency or an extra only ever
means editing that file.

Usage:
    python scripts/build_core_package.py [--outdir dist]

    # rewrite pyproject.toml in place, to install `mteb-core` from the working tree (used by CI)
    python scripts/build_core_package.py --edit-pyproject
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

# left out of `mteb-core`; only needed to load and run models, so only in a full `mteb` install
FULL_INSTALL_DEPENDENCIES = {"torch", "transformers", "sentence-transformers"}


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
        if canonicalize_name(Requirement(dependency).name) in FULL_INSTALL_DEPENDENCIES
    ]
    found = {canonicalize_name(Requirement(dependency).name) for dependency in run}
    if found != FULL_INSTALL_DEPENDENCIES:
        raise ValueError(
            f"dependencies {FULL_INSTALL_DEPENDENCIES - found} are missing from pyproject.toml"
        )

    # an extra requiring `mteb[...]` would make `mteb-core` depend on `mteb`, installing the same files twice
    self_referencing = [
        f"{extra}: {dependency}"
        for extra, dependencies in project["optional-dependencies"].items()
        for dependency in dependencies
        if canonicalize_name(Requirement(dependency).name) == "mteb"
    ]
    if self_referencing:
        raise ValueError(f"extras must not require `mteb` itself: {self_referencing}")

    core = pyproject.replace('\nname = "mteb"\n', '\nname = "mteb-core"\n', 1)
    for dependency in run:
        line = f'    "{dependency}",\n'
        if line not in core:
            raise ValueError(
                f"expected {line.strip()!r} on its own line in dependencies"
            )
        core = core.replace(line, "", 1)
    return core


def build(source: Path, outdir: Path) -> None:
    """Build the sdist and wheel of `source` into `outdir`, in an isolated build environment."""
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(outdir.resolve()), str(source)],
        check=True,
    )


def build_core_package(outdir: Path) -> None:
    """Build `mteb-core` from the working tree into `outdir`."""
    # the build only sees what is copied below, so a packaging file added later would be ignored silently
    unsupported = [
        f for f in ("MANIFEST.in", "setup.py", "setup.cfg") if (REPO_ROOT / f).exists()
    ]
    if unsupported:
        raise ValueError(
            f"{unsupported} are not copied into the `mteb-core` build; update this script"
        )

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
    parser.add_argument(
        "--edit-pyproject",
        action="store_true",
        help="turn the working tree into `mteb-core` by rewriting pyproject.toml, instead of building",
    )
    args = parser.parse_args()

    pyproject = REPO_ROOT / "pyproject.toml"
    if args.edit_pyproject:
        pyproject.write_text(
            core_pyproject(pyproject.read_text(encoding="utf-8")), encoding="utf-8"
        )
        return
    build_core_package(args.outdir)


if __name__ == "__main__":
    main()

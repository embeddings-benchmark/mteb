"""Compare installing `mteb` with installing `mteb-core` (download size, install time, installed size, import time).

Both wheels are built from the working tree. Every install uses a fresh virtual environment and an empty cache,
so install times include downloading, and depend on the network.

The Linux download size is computed without installing: both wheels are resolved for Linux x86_64 and the sizes
of the resolved wheels are read from PyPI. On Linux the default torch wheel pulls in the CUDA libraries.

Requires `uv` on the PATH. Usage:
    uv run --no-project --python 3.12 --with build --with packaging python scripts/benchmark_core_package.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = {"mteb": "mteb", "mteb-core": "mteb_core"}  # name -> wheel file prefix


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True, **kwargs)


def build_wheels(outdir: Path) -> dict[str, Path]:
    run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(outdir),
            str(REPO_ROOT),
        ]
    )
    run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/build_core_package.py"),
            "--outdir",
            str(outdir),
        ]
    )
    return {
        name: next(outdir.glob(f"{prefix}-*.whl")) for name, prefix in PACKAGES.items()
    }


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def install(
    wheel: Path, tool: str, workdir: Path, python_version: str
) -> dict[str, Any]:
    """Install `wheel` into a fresh venv with an empty cache and return the timing and size."""
    venv = workdir / f"venv-{wheel.name}-{tool}"
    run(["uv", "venv", "--seed", "--python", python_version, str(venv)])
    python = venv_python(venv)
    if tool == "uv":
        command = ["uv", "pip", "install", "--python", str(python), str(wheel)]
        env = {**os.environ, "UV_CACHE_DIR": str(workdir / f"cache-{wheel.name}")}
    else:
        command = [str(python), "-m", "pip", "install", "--no-cache-dir", str(wheel)]
        env = None
    start = time.perf_counter()
    run(command, env=env)
    seconds = time.perf_counter() - start

    listed = run(
        ["uv", "pip", "list", "--python", str(python), "--format", "json"]
    ).stdout
    site_packages = Path(
        run(
            [
                str(python),
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ]
        ).stdout.strip()
    )
    size = sum(f.stat().st_size for f in site_packages.rglob("*") if f.is_file())
    return {
        "seconds": seconds,
        "packages": len(json.loads(listed)),
        "mb": size / 1e6,
        "python": python,
    }


def import_time(python: Path, repeats: int) -> tuple[float, bool]:
    """Best `import mteb` time over `repeats` fresh interpreters, and whether torch was imported."""
    script = "import sys, time; t = time.perf_counter(); import mteb; print(time.perf_counter() - t, 'torch' in sys.modules)"
    # run outside the repository, so the installed package is imported instead of the working tree
    results = [
        run([str(python), "-c", script], cwd=tempfile.gettempdir()).stdout.split()
        for _ in range(repeats)
    ]
    return min(float(seconds) for seconds, _ in results), results[0][1] == "True"


def linux_download_size(wheel: Path, python_version: str) -> tuple[int, float]:
    """Number of dependencies and total download size (GB) of `wheel` on Linux x86_64, read from PyPI."""
    resolved = run(
        [
            "uv",
            "pip",
            "compile",
            "--quiet",
            "--no-header",
            "--no-annotate",
            "--python-version",
            python_version,
            "--python-platform",
            "x86_64-manylinux_2_28",
            "-",
        ],
        input=str(wheel),
    ).stdout
    pins = [
        line.split("==")
        for line in resolved.splitlines()
        if "==" in line and not line.startswith("#")
    ]
    tag = f"cp{python_version.replace('.', '')}"

    def size(pin: list[str]) -> int:
        name, version = pin[0].strip(), pin[1].split(";")[0].strip()
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{name}/{version}/json"
        ) as response:
            files = json.load(response)["urls"]
        wheels = [
            f["size"]
            for f in files
            if f["packagetype"] == "bdist_wheel"
            and re.search(rf"({tag}|py3|abi3)", f["filename"])
            and re.search(r"(manylinux.*x86_64|none-any)", f["filename"])
        ]
        return max(wheels, default=0)

    with ThreadPoolExecutor(16) as executor:
        return len(pins), sum(executor.map(size, pins)) / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="number of `import mteb` runs per install",
    )
    args = parser.parse_args()

    rows = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        wheels = build_wheels(workdir / "dist")
        for name, wheel in wheels.items():
            print(f"measuring {name} ...", file=sys.stderr)
            uv = install(wheel, "uv", workdir, args.python_version)
            pip = install(wheel, "pip", workdir, args.python_version)
            seconds, torch_imported = import_time(uv["python"], args.repeats)
            n_linux, gb_linux = linux_download_size(wheel, args.python_version)
            rows[name] = [
                f"{gb_linux:.2f} GB ({n_linux} packages)",
                f"{pip['seconds']:.0f} s / {uv['seconds']:.0f} s",
                f"{pip['mb']:.0f} MB ({pip['packages']} packages)",
                f"{seconds:.2f} s (torch imported: {torch_imported})",
            ]

    labels = [
        "Download on Linux x86_64",
        f"Install time, pip / uv ({sys.platform})",
        f"Installed size ({sys.platform})",
        "`import mteb` (best of runs)",
    ]
    print(f"\nPython {args.python_version}, fresh venv and empty cache per install\n")
    print("| | " + " | ".join(f"`{name}`" for name in rows) + " |")
    print("|---|" + "---|" * len(rows))
    for i, label in enumerate(labels):
        print(
            f"| {label} | " + " | ".join(values[i] for values in rows.values()) + " |"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util


def _is_package_available(pkg_name: str) -> bool:
    package_exists = importlib.util.find_spec(pkg_name) is not None
    return package_exists


def requires_package(
    obj, package_name: str, model_name: str, install_instruction: str | None = None
) -> None:
    if not _is_package_available(package_name):
        install_instruction = (
            f"pip install {package_name}"
            if install_instruction is None
            else install_instruction
        )
        name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
        raise ImportError(
            f"{name} requires the `{package_name}` library but it was not found in your environment. "
            + f"If you want to load {model_name} models, please `{install_instruction}` to install the package."
        )

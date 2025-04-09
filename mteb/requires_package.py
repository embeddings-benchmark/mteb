from __future__ import annotations

import importlib.util
import logging

logger = logging.getLogger(__name__)


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


def suggest_package(
    obj, package_name: str, model_name: str, install_instruction: str
) -> bool:
    """Check if a package is available and log a warning with installation instructions if it's not.
    Unlike requires_package, this doesn't raise an error but returns True if the package is available.
    """
    if not _is_package_available(package_name):
        name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
        logger.warning(
            f"{name} can benefit from the `{package_name}` library but it was not found in your environment. "
            + f"{model_name} models were trained with flash attention enabled. For optimal performance, please install the `{package_name}` package with `{install_instruction}`."
        )
        return False
    return True


def requires_image_dependencies() -> None:
    if not _is_package_available("torchvision"):
        raise ImportError(
            "You are trying to running the image subset of mteb without having installed the required dependencies (`torchvision`). "
            + "You can install the required dependencies using `pip install 'mteb[image]'` to install the required dependencies."
        )

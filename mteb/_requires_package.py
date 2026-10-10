from __future__ import annotations

import functools
import importlib.metadata
import importlib.util
import logging
from typing import TYPE_CHECKING

from typing_extensions import deprecated

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@functools.cache
def _mteb_distribution() -> importlib.metadata.Distribution:
    """Return the installed distribution that provides the `mteb` package.

    The code is published both as `mteb` and as `mteb-core`, a minimal install of the same code without the
    dependencies needed to load and run models. `mteb-core` is checked first, so that the install instructions
    say to switch to `mteb`.
    """
    for name in ("mteb-core", "mteb"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        if name == "mteb-core" and _is_installed("mteb"):
            logger.warning(
                "`mteb` and `mteb-core` are both installed. They contain the same files, so uninstalling "
                "either one removes the files of the other. Uninstall both, then install only one of them."
            )
        return distribution
    raise importlib.metadata.PackageNotFoundError("mteb")


def _is_installed(distribution_name: str) -> bool:
    try:
        importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _extras_requirement(groups: Sequence[str]) -> str:
    """The requirement that installs the given extras groups for the installed distribution."""
    return f"{_mteb_distribution().metadata['Name']}[{','.join(groups)}]"


def _install_command(groups: Sequence[str] = ()) -> str:
    """The command to install the given extras groups, or a full mteb installation if there are none.

    A full installation means `mteb`, so on `mteb-core` it means switching. Both contain the same files, so
    `mteb-core` has to be uninstalled first, rather than leaving two distributions owning them.
    """
    if groups:
        return f"pip install {_extras_requirement(groups)}"
    if _mteb_distribution().metadata["Name"] == "mteb-core":
        return "pip uninstall -y mteb-core && pip install mteb"
    return "pip install mteb"


def _full_installation_message(what: str) -> str:
    """The error message for something that a minimal `mteb-core` installation cannot do."""
    return f"{what} requires a full mteb installation. Install it with `{_install_command()}`."


def _requires_full_installation(package: str, what: str) -> None:
    """Raise an error saying what to install if this is a minimal `mteb-core` installation.

    Args:
        package: The package to check, one of the dependencies that `mteb-core` leaves out.
        what: What the user was doing, e.g. "Evaluating a model".
    """
    if not _is_package_available(package):
        raise ImportError(_full_installation_message(what))


def _is_package_available(pkg_name: str) -> bool:
    package_exists = importlib.util.find_spec(pkg_name) is not None
    return package_exists


@deprecated(
    "Use ModelMeta.extra_requirements_groups instead of requires_package. This function will be removed in a future version."
)
def requires_package(
    obj: object,
    package_name: str,
    model_name: str,
    install_instruction: str | None = None,
) -> None:
    """Check if a package is available and raise an error with installation instructions if it's not.

    Args:
        obj: The object (class or function) that requires the package.
        package_name: The name of the package to check.
        model_name: The name of the model that benefits from the package.
        install_instruction: The instruction to install the package. If None, defaults to "pip install {package_name}".
    """
    if not _is_package_available(package_name):
        install_instruction = (
            f"pip install {package_name}"
            if install_instruction is None
            else install_instruction
        )
        name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
        raise ImportError(
            f"{name} requires the `{package_name}` library but it was not found in your environment. "
            f"If you want to load {model_name} models, please `{install_instruction}` to install the package."
        )


def suggest_package(
    obj: object, package_name: str, model_name: str, install_instruction: str
) -> bool:
    """Suggestion to install a package.

    Check if a package is available and log a warning with installation instructions if it's not.
    Unlike requires_package, this doesn't raise an error but returns True if the package is available.

    Args:
        obj: The object (class or function) that requires the package.
        package_name: The name of the package to check.
        model_name: The name of the model that benefits from the package.
        install_instruction: The instruction to install the package.

    Returns:
        bool: True if the package is available, False otherwise.
    """
    if not _is_package_available(package_name):
        name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
        logger.warning(
            f"{name} can benefit from the `{package_name}` library but it was not found in your environment. "
            f"{model_name} models were trained with flash attention enabled. For optimal performance, please install the `{package_name}` package with `{install_instruction}`."
        )
        return False
    return True


@deprecated(
    "Use ModelMeta.extra_requirements_groups instead of requires_package. This function will be removed in a future version."
)
def requires_image_dependencies() -> None:
    """Check if the required dependencies for image tasks are available."""
    if not _is_package_available("torchvision"):
        raise ImportError(
            "You are trying to running the image subset of mteb without having installed the required dependencies (`torchvision`). "
            "You can install the required dependencies using `pip install 'mteb[image]'` to install the required dependencies."
        )


@deprecated(
    "Use ModelMeta.extra_requirements_groups instead of requires_package. This function will be removed in a future version."
)
def requires_audio_dependencies() -> None:
    """Check if the required dependencies for audio tasks are available."""
    if not _is_package_available("torchaudio"):
        raise ImportError(
            "You are trying to running the audio subset of mteb without having installed the required dependencies (`torchaudio`). "
            "You can install the required dependencies using `pip install 'mteb[audio]'` to install the required dependencies."
        )

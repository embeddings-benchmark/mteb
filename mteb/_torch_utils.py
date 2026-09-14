"""Torch helpers that do not import torch until they are used.

Model implementations are imported when `mteb` builds its model registry, including on installs
without torch, so anything they evaluate at import time must not need torch. That rules out
`@torch.no_grad()` / `@torch.inference_mode()` decorators and `torch.cuda.is_available()` default
arguments; these helpers do the same work when called instead.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
R = TypeVar("R")


def no_grad(func: Callable[P, R]) -> Callable[P, R]:
    """Equivalent to `@torch.no_grad()`, applied when `func` is called rather than defined.

    Torch's own decorator is applied at call time, so its behaviour carries over exactly --
    including generator functions, where the context is held around each step.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        import torch

        return torch.no_grad()(func)(*args, **kwargs)

    return wrapper


def inference_mode(func: Callable[P, R]) -> Callable[P, R]:
    """Equivalent to `@torch.inference_mode()`, applied when `func` is called rather than defined.

    Torch's own decorator is applied at call time, so its behaviour carries over exactly --
    including generator functions, where the context is held around each step.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        import torch

        return torch.inference_mode()(func)(*args, **kwargs)

    return wrapper


def get_device(device: str | None = None) -> str:
    """Return `device` if given, otherwise the best available of `"cuda"`, `"mps"` and `"cpu"`."""
    if device is not None:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

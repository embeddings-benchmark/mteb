"""Guards that keep heavy runtime dependencies out of the import path.

Analysis-only users (results, leaderboard, API) should not pay for torch. These tests pin the
modules that have already been made torch-free so the eager imports do not creep back in.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# `mteb/__init__.py` still pulls torch in via `mteb.models` and `mteb.abstasks`, so a plain
# `import mteb.types` would import it transitively. Stubbing the package in `sys.modules` lets us
# import the submodule on its own. Once those two are lazy as well this can become `import mteb`.
_IMPORT_TYPES_WITHOUT_PACKAGE_INIT = """
import pathlib
import sys
import types

pkg = types.ModuleType("mteb")
pkg.__path__ = [str(pathlib.Path("mteb").resolve())]
sys.modules["mteb"] = pkg

import mteb.types  # noqa: F401
"""


def _run(extra: str) -> str:
    """Import `mteb.types` in a fresh interpreter, then run `extra` and return its stdout."""
    script = _IMPORT_TYPES_WITHOUT_PACKAGE_INIT + textwrap.dedent(extra)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_importing_types_does_not_import_torch() -> None:
    """`mteb.types` must stay importable without torch."""
    loaded = _run(
        """
        print("torch" in sys.modules)
        """
    )
    assert loaded == "False", (
        "importing mteb.types pulled in torch; keep torch under `TYPE_CHECKING` "
        "or move it into the function that needs it"
    )


def test_array_is_still_importable_at_runtime() -> None:
    """`from mteb.types import Array` is public and documented, so it must not need torch.

    The `torch.Tensor` arm of the alias only exists for type checkers; the runtime value is the
    numpy-only fallback, which is fine because `Array` is only ever used as an annotation.
    """
    output = _run(
        """
        import numpy as np
        from numpy.typing import NDArray

        from mteb.types import Array

        print(Array == NDArray[np.floating | np.integer | np.bool_])
        print("torch" in sys.modules)
        """
    )
    assert output.splitlines() == ["True", "False"]

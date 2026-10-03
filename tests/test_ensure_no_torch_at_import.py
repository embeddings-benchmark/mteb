"""Guards that keep torch out of `import mteb`.

`mteb-core` is a minimal install of mteb, without torch, transformers and sentence-transformers. The `test-core`
CI job runs the tests that must pass on it; these guards also run on a full installation, where an eager import
or a torch dtype evaluated at import time would otherwise go unnoticed.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

import pytest

from mteb.models.model_meta import _FULL_INSTALL_DEPENDENCIES

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _run(script: str) -> str:
    """Run `script` in a fresh interpreter and return the last line it printed."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout.strip().splitlines()[-1]


def test_import_mteb_does_not_import_torch() -> None:
    """Even when torch is installed, `import mteb` and the CLI must not import it."""
    loaded = _run(
        f"""
        import sys
        import mteb
        import mteb.cli

        print("loaded:", ",".join(d for d in {sorted(_FULL_INSTALL_DEPENDENCIES)!r} if d in sys.modules))
        """
    )
    assert loaded == "loaded:", (
        f"`import mteb` {loaded}; import it inside the function that uses it, or under "
        "`TYPE_CHECKING` if it is only used in type hints"
    )


def test_missing_full_installation_says_what_to_install(monkeypatch) -> None:
    """On `mteb-core`, anything that needs torch must say how to get a full installation."""
    import mteb._requires_package as requires_package

    monkeypatch.setattr(requires_package, "_is_package_available", lambda name: False)
    with pytest.raises(ImportError, match="Install it with `pip .*install mteb`"):
        requires_package._requires_full_installation("torch", "Evaluating a model")


@pytest.mark.full_install
def test_model_implementations_declare_no_import_time_torch_dtypes() -> None:
    """No model file may evaluate a `torch.<dtype>` at import time; use `OutputDType` instead.

    Everything outside a function body runs at import -- module-level `ModelMeta(...)` calls,
    class bodies, decorators and signature defaults -- so only function bodies are exempt.
    """
    import ast

    import torch

    # every torch attribute that is itself a dtype, e.g. "float32", "bfloat16", but also
    # aliases like "float", "half" and "long" that OutputDType has no member for
    dtypes = {
        name for name in dir(torch) if isinstance(getattr(torch, name), torch.dtype)
    }

    offenders = []
    for path in sorted(
        (_REPO_ROOT / "mteb/models/model_implementations").rglob("*.py")
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # nodes actually inside a function/method body are evaluated lazily and exempt;
        # matched by identity (not line number) so a one-line `def f(x=torch.bfloat16): ...`
        # doesn't let the signature default hide behind its body's line number
        in_body = {
            id(sub)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for stmt in node.body
            for sub in ast.walk(stmt)
        }
        offenders += [
            f"{path.name}:{node.lineno} torch.{node.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "torch"
            and node.attr in dtypes
            and id(node) not in in_body
        ]

    assert not offenders, (
        "these evaluate a torch dtype at import time; use the matching OutputDType member "
        f"instead: {offenders}"
    )

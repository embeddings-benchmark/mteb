"""Guards that keep torch out of `import mteb`.

`mteb-core` is mteb without torch, transformers and sentence-transformers (they are in its `run` extra), so working
with tasks, benchmarks, model metadata and results must not need them.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

_HEAVY = ("torch", "transformers", "sentence_transformers")

# Hides packages from the import system as if they were not installed: `import torch` fails and
# `importlib.util.find_spec("torch")` returns None. (Setting `sys.modules["torch"] = None` is not
# faithful -- libraries such as scipy see the key and dereference it.)
_WITHOUT_TORCH = f"""
import importlib.machinery
import sys


class _Hide(importlib.machinery.PathFinder):
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname.split(".")[0] in {set(_HEAVY)!r}:
            return None
        return super().find_spec(fullname, path, target)


sys.meta_path = [_Hide if f is importlib.machinery.PathFinder else f for f in sys.meta_path]
"""


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

        print("loaded:", ",".join(d for d in {_HEAVY!r} if d in sys.modules))
        """
    )
    assert loaded == "loaded:", (
        f"`import mteb` {loaded}; import it inside the function that uses it, or under "
        "`TYPE_CHECKING` if it is only used in type hints"
    )


def test_mteb_works_without_torch_installed() -> None:
    """Tasks, the CLI and every model's metadata must work on an install without torch.

    Building the model registry imports all model implementation files, so any of them importing
    torch, transformers or sentence-transformers at module scope fails this test. Loading a model
    must say what to install.
    """
    output = _run(
        _WITHOUT_TORCH
        + textwrap.dedent(
            """
            import mteb
            import mteb.cli

            assert len(mteb.get_model_metas()) > 0
            assert len(mteb.get_tasks(tasks=["NFCorpus"])) == 1
            try:
                mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")
            except ModuleNotFoundError as e:
                print(e)
            """
        )
    )
    assert "To load and run models, install `" in output


def test_set_seed_still_seeds_torch_when_available() -> None:
    """Torch must still be seeded once it is imported."""
    import torch

    from mteb._set_seed import _set_seed

    _set_seed(42)
    first = torch.randn(4)
    _set_seed(42)
    second = torch.randn(4)
    assert torch.equal(first, second)


def test_image_dataset_works_in_dataloader_worker_processes() -> None:
    """With `num_proc > 1` the image dataset and collate fn are pickled into worker processes.

    `spawn` (the default on macOS and Windows) cannot pickle anything defined inside a function.
    """
    from PIL import Image
    from torch.utils.data import DataLoader

    from mteb._evaluators.image.imagetext_pairclassification_evaluator import (
        _build_image_dataset,
        _image_collate_fn,
    )

    loader = DataLoader(
        _build_image_dataset([Image.new("RGB", (8, 8)) for _ in range(3)]),
        batch_size=2,
        num_workers=2,
        collate_fn=_image_collate_fn,
        multiprocessing_context="spawn",
    )
    assert [len(batch["image"]) for batch in loader] == [2, 1]


def test_mteb_distribution_finds_mteb_or_mteb_core(monkeypatch, caplog) -> None:
    """The version, extras and requirements are read from whichever of `mteb` and `mteb-core` is installed.

    Both ship the same files, so installing them together is a mistake worth warning about: uninstalling
    either one then removes the files of the other.
    """
    import importlib.metadata

    from mteb._requires_package import _mteb_distribution

    def installed(*names: str):
        def distribution(name: str) -> str:
            if name in names:
                return name
            raise importlib.metadata.PackageNotFoundError(name)

        return distribution

    cases = [
        (("mteb",), "mteb", False),  # pip install mteb
        (("mteb-core",), "mteb-core", False),  # pip install mteb-core
        (
            ("mteb", "mteb-core"),
            "mteb-core",
            True,
        ),  # both, which breaks on the next uninstall
    ]
    try:
        for names, expected, warns in cases:
            monkeypatch.setattr(importlib.metadata, "distribution", installed(*names))
            _mteb_distribution.cache_clear()
            caplog.clear()
            with caplog.at_level("WARNING"):
                assert _mteb_distribution() == expected
            assert ("both installed" in caplog.text) is warns, caplog.text
    finally:
        _mteb_distribution.cache_clear()


def test_model_meta_dtypes_are_named_not_torch_objects() -> None:
    """`ModelMeta` declares load dtypes by name, so metadata stays importable without torch.

    `OutputDType` is a `str` enum, so it resolves through both `getattr(torch, ...)` (the path
    transformers takes for a string dtype) and `OutputDType.get_dtype()`.
    """
    import torch

    import mteb
    from mteb.types import OutputDType

    meta = mteb.get_model_meta("vidore/colpali-v1.1")
    declared = meta.loader_kwargs["torch_dtype"]

    assert isinstance(declared, OutputDType)
    assert declared.get_dtype() is torch.float16
    assert getattr(torch, declared) is torch.float16


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

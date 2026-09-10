"""Guards that keep heavy runtime dependencies out of the import path.

Analysis-only users (results, leaderboard, API) should not pay for torch. These tests pin the
modules that have already been made torch-free so the eager imports do not creep back in.
"""

from __future__ import annotations

import pathlib
import random
import subprocess
import sys
import textwrap

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# `mteb/__init__.py` (and `mteb/models/__init__.py`) still pull torch in, so importing a submodule
# normally drags it along transitively. Stubbing the parent packages in `sys.modules` lets us
# import the submodule on its own. Once those packages are lazy too this can become `import mteb`.
_PREAMBLE = """
import pathlib
import sys
import types

_root = pathlib.Path({root!r})
sys.path.insert(0, str(_root))
for _name in {stubs!r}:
    _stub = types.ModuleType(_name)
    _stub.__path__ = [str(_root.joinpath(*_name.split(".")))]
    sys.modules[_name] = _stub

import {module}  # noqa: F401
"""

_HEAVY = ("torch", "sentence_transformers", "transformers")


def _run(module: str, stubs: tuple[str, ...], extra: str = "") -> str:
    """Import `module` in a fresh interpreter with `stubs` faked, then run `extra`."""
    script = _PREAMBLE.format(root=str(_REPO_ROOT), stubs=stubs, module=module)
    result = subprocess.run(
        [sys.executable, "-c", script + textwrap.dedent(extra)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _loaded_heavy_deps(module: str, stubs: tuple[str, ...]) -> list[str]:
    out = _run(
        module,
        stubs,
        f"""
        print(",".join(d for d in {_HEAVY!r} if d in sys.modules))
        """,
    )
    return [d for d in out.split(",") if d]


def test_importing_types_does_not_import_torch() -> None:
    """`mteb.types` must stay importable without torch."""
    assert _loaded_heavy_deps("mteb.types", ("mteb",)) == [], (
        "importing mteb.types pulled in a heavy dependency; keep it under `TYPE_CHECKING` "
        "or move it into the function that needs it"
    )


def test_array_is_still_importable_at_runtime() -> None:
    """`from mteb.types import Array` is public and documented, so it must not need torch.

    The `torch.Tensor` arm of the alias only exists for type checkers; the runtime value is the
    numpy-only fallback, which is fine because `Array` is only ever used as an annotation.
    """
    output = _run(
        "mteb.types",
        ("mteb",),
        """
        import numpy as np
        from numpy.typing import NDArray

        from mteb.types import Array

        print(Array == NDArray[np.floating | np.integer | np.bool_])
        print("torch" in sys.modules)
        """,
    )
    assert output.splitlines() == ["True", "False"]


def test_importing_model_meta_does_not_import_torch() -> None:
    """`ModelMeta` is pure metadata; describing a model must not require loading one."""
    assert (
        _loaded_heavy_deps("mteb.models.model_meta", ("mteb", "mteb.models")) == []
    ), (
        "importing mteb.models.model_meta pulled in a heavy dependency; keep it under "
        "`TYPE_CHECKING` or move it into the function that needs it"
    )


def test_model_meta_is_usable_without_torch() -> None:
    """Constructing and serialising a ModelMeta must work on a torch-free install."""
    output = _run(
        "mteb.models.model_meta",
        ("mteb", "mteb.models"),
        """
        from mteb.models.model_meta import ModelMeta

        meta = ModelMeta(
            loader=None,
            name="test/model",
            revision="abc123",
            release_date="2024-01-01",
            languages=["eng-Latn"],
            n_parameters=1000,
            memory_usage_mb=1.0,
            max_tokens=512,
            embed_dim=64,
            license="mit",
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            similarity_fn_name="cosine",
            framework=["Sentence Transformers"],
            reference=None,
            use_instructions=False,
            training_datasets=None,
        )
        print(meta.name, meta.revision)
        print("torch" in sys.modules)
        """,
    )
    assert output.splitlines() == ["test/model abc123", "False"]


def test_importing_set_seed_does_not_import_torch() -> None:
    """`AbsTask.__init__` seeds on construction, so this runs on the metadata-only path."""
    assert _loaded_heavy_deps("mteb._set_seed", ("mteb",)) == [], (
        "importing mteb._set_seed pulled in a heavy dependency; import torch inside the "
        "function that needs it"
    )


def test_set_seed_still_seeds_torch_when_available() -> None:
    """The lazy import must not stop torch actually being seeded on a full install."""
    import torch

    from mteb._set_seed import _set_seed

    _set_seed(42)
    first = torch.randn(4)
    _set_seed(42)
    second = torch.randn(4)
    assert torch.equal(first, second)


def test_set_seed_works_without_torch(monkeypatch) -> None:
    """On a torch-free install, seeding must degrade rather than raise."""
    import mteb._set_seed as seed_module

    monkeypatch.setattr(
        seed_module, "_is_package_available", lambda name: name != "torch"
    )
    rng, np_rng = seed_module._set_seed(42)
    assert rng.random() == random.Random(42).random()
    assert np_rng is not None


def test_importing_create_dataloaders_does_not_import_torch() -> None:
    """Dataloaders are only built when a task actually runs, so importing must not need torch."""
    assert _loaded_heavy_deps("mteb._create_dataloaders", ("mteb",)) == [], (
        "importing mteb._create_dataloaders pulled in a heavy dependency; move the import "
        "into the function that uses it"
    )


def test_image_dataset_builder_still_returns_a_torch_dataset() -> None:
    """`CustomImageDataset` moved inside a function; it must still behave identically."""
    from PIL import Image as PILImage
    from torch.utils.data import DataLoader, Dataset

    from mteb._evaluators.image.imagetext_pairclassification_evaluator import (
        _build_image_dataset,
    )

    images = [PILImage.new("RGB", (8, 8)) for _ in range(3)]
    dataset = _build_image_dataset(images)

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 3
    assert dataset.features == {"image": []}
    assert list(dataset[0]) == ["image"]

    loader = DataLoader(
        dataset, batch_size=2, collate_fn=lambda b: {"image": [i["image"] for i in b]}
    )
    assert [len(batch["image"]) for batch in loader] == [2, 1]


def test_importing_abstasks_does_not_import_torch() -> None:
    """Task classes carry the metadata the leaderboard and API read; describing them needs no torch."""
    assert _loaded_heavy_deps("mteb.abstasks", ("mteb",)) == [], (
        "importing mteb.abstasks pulled in a heavy dependency; move the import into the "
        "method that uses it"
    )


def test_importing_benchmarks_does_not_import_torch() -> None:
    """Benchmark definitions are metadata over tasks, so they must not need torch either."""
    assert _loaded_heavy_deps("mteb.benchmarks.benchmark", ("mteb",)) == [], (
        "importing mteb.benchmarks.benchmark pulled in a heavy dependency"
    )


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
    """No model file may evaluate a `torch.<dtype>` at import time; use `OutputDType` instead."""
    import ast

    dtypes = {"float16", "float32", "bfloat16", "uint8", "int8", "float64"}
    offenders = []
    for path in sorted(
        (_REPO_ROOT / "mteb/models/model_implementations").rglob("*.py")
    ):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # only signature defaults execute at import; bodies are evaluated lazily
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if default is None:
                    continue
                for sub in ast.walk(default):
                    if (
                        isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id == "torch"
                        and sub.attr in dtypes
                    ):
                        offenders.append(f"{path.name}:{sub.lineno} torch.{sub.attr}")

    assert not offenders, (
        "these default arguments evaluate a torch dtype at import time; use the matching "
        f"OutputDType member instead: {offenders}"
    )

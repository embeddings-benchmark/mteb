"""Guard against heavy runtime imports on the model-metadata hot path.

Building MODEL_REGISTRY (and by extension running scripts like
generate_leaderboard_models.py) only needs model metadata — it must not
force the `datasets` / `pyarrow` stack to be loaded. On GitHub Actions
runners that import alone can add 10-20 min to cold-start time.

Two complementary approaches are used:

* **AST (static)** — parse source files with the `ast` module and verify
  that banned imports only appear inside ``if TYPE_CHECKING:`` blocks.
  Zero Python imports required; runs in milliseconds.

* **Subprocess (runtime)** — import only the *lightweight* public modules
  (``mteb.types``) in a fresh interpreter and assert the banned library is
  absent from ``sys.modules``. Targets only modules that are cheap to
  import so the test suite stays fast.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _top_level_imported_packages(rel_path: str) -> set[str]:
    """Return top-level package names that are *unconditionally* imported.

    "Unconditional" means the import statement sits directly in the module
    body — not inside an ``if TYPE_CHECKING:`` guard (or any other ``if``
    block).  Only the root package name is returned, e.g. ``"datasets"``
    for ``from datasets import Dataset``.
    """
    source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=rel_path)

    packages: set[str] = set()
    for node in tree.body:  # walk only the module's top-level statements
        if isinstance(node, ast.ImportFrom) and node.module:
            packages.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                packages.add(alias.name.split(".")[0])
    return packages


def _subprocess_import_check(script: str, banned_package: str) -> str | None:
    """Write *script* to a temp file, run it, assert *banned_package* is absent.

    Returns an error string on failure, None on success.
    """
    import tempfile

    check_lines = (
        "import sys\n"
        f"leaked = [k for k in sys.modules if k == '{banned_package}' or k.startswith('{banned_package}.')]\n"
        "assert not leaked, repr(leaked)\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".py", delete=False
    ) as f:
        f.write(script + "\n" + check_lines)
        tmp_path = f.name
    result = subprocess.run(
        [sys.executable, tmp_path],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    Path(tmp_path).unlink(missing_ok=True)
    return result.stderr.strip() if result.returncode != 0 else None


# ---------------------------------------------------------------------------
# AST-based tests (static, no imports required)
# ---------------------------------------------------------------------------


def test_encoder_io_no_unconditional_datasets_import() -> None:
    """``mteb/types/_encoder_io.py`` must not import ``datasets`` at the top level.

    _encoder_io.py is pulled in by every ModelMeta construction via
    ``mteb.types``.  A bare ``from datasets import Dataset`` there forces
    datasets+pyarrow into every process that merely reads model metadata.
    The import must be guarded with ``if TYPE_CHECKING:``.
    """
    packages = _top_level_imported_packages("mteb/types/_encoder_io.py")
    assert "datasets" not in packages, (
        "mteb/types/_encoder_io.py imports 'datasets' unconditionally. "
        "Move `from datasets import Dataset` inside `if TYPE_CHECKING:` and "
        "use a string TypeAlias (e.g. QueryDatasetType: TypeAlias = 'Dataset') "
        "to avoid loading datasets+pyarrow when model metadata is read."
    )


def test_model_files_no_unconditional_create_dataloaders_import() -> None:
    """Model implementation files must not import ``_create_dataloaders`` at the top level.

    ``_create_dataloaders`` imports ``datasets`` unconditionally.  When model
    implementation files import it at module scope, building MODEL_REGISTRY
    (which imports all ~238 model files) pulls in datasets+pyarrow as a
    side-effect — even though those libraries are only needed at inference
    time.  The imports must be deferred to the methods/functions that use them.
    """
    # Only files that were identified as offenders in the Aug-18 regression.
    files_under_test = [
        "mteb/models/model_implementations/bm25.py",
        "mteb/models/model_implementations/bb25.py",
        "mteb/models/model_implementations/random_baseline.py",
        "mteb/models/model_implementations/pylate_models.py",
        "mteb/models/model_implementations/cde_models.py",
    ]

    offenders = []
    for rel_path in files_under_test:
        # _create_dataloaders is the internal module; its root package is "mteb"
        # so we inspect the import nodes more specifically rather than using
        # _top_level_imported_packages (which only returns root package names).
        source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel_path)
        for node in tree.body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and "_create_dataloaders" in node.module
            ):
                offenders.append(rel_path)
                break

    assert not offenders, (
        "The following model files import `_create_dataloaders` unconditionally "
        "(which transitively loads datasets+pyarrow at registry-build time):\n"
        + "\n".join(f"  {p}" for p in offenders)
        + "\nMove these imports inside the methods/functions that use them."
    )


# ---------------------------------------------------------------------------
# Subprocess-based tests (runtime, lightweight modules only)
# ---------------------------------------------------------------------------


def test_encoder_io_does_not_import_datasets_at_runtime() -> None:
    """Loading ``mteb/types/_encoder_io.py`` at runtime must not import ``datasets``.

    Uses a subprocess that loads the file in isolation (bypassing
    ``mteb/__init__.py``) so unrelated imports in the rest of the package
    cannot mask a regression here.
    """
    script = """\
import sys, types as _t, importlib.util

# Stub parent packages so Python skips their __init__.py files.
for _pkg in ("mteb", "mteb.types", "mteb._helpful_enum"):
    sys.modules.setdefault(_pkg, _t.ModuleType(_pkg))

# _encoder_io subclasses HelpfulStrEnum; stub it as plain str so that
# PromptType / OutputDType can be defined without enum machinery.
sys.modules["mteb._helpful_enum"].HelpfulStrEnum = str

# Load _encoder_io via importlib without triggering any __init__.py.
spec = importlib.util.spec_from_file_location(
    "mteb.types._encoder_io", "mteb/types/_encoder_io.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
"""
    err = _subprocess_import_check(script, "datasets")
    assert err is None, err

import os
import sys
from importlib.metadata import version

from mteb import types
from mteb.abstasks import AbsTask
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.results import BenchmarkResults, TaskResult

__version__ = version("mteb")  # fetch version from install metadata

__all__ = [
    "MTEB",
    "AbsTask",
    "Benchmark",
    "BenchmarkResults",
    "CacheBackendProtocol",
    "CrossEncoderProtocol",
    "EncoderProtocol",
    "IndexEncoderSearchProtocol",
    "MTEBTasks",
    "SearchProtocol",
    "SentenceTransformerEncoderWrapper",
    "TaskMetadata",
    "TaskResult",
    "evaluate",
    "filter_tasks",
    "get_benchmark",
    "get_benchmarks",
    "get_model",
    "get_model_meta",
    "get_model_metas",
    "get_task",
    "get_tasks",
    "load_results",
    "types",
]


def __getattr__(attr_name):
    """Lazy import heavy dependencies only when needed."""
    import importlib

    _module_map = {
        "MTEB": ("mteb.deprecated_evaluator", "MTEB"),
        "evaluate": ("mteb.evaluate", "evaluate"),
        "filter_tasks": ("mteb.filter_tasks", "filter_tasks"),
        "get_task": ("mteb.get_tasks", "get_task"),
        "get_tasks": ("mteb.get_tasks", "get_tasks"),
        "MTEBTasks": ("mteb.get_tasks", "MTEBTasks"),
        "load_results": ("mteb.load_results", "load_results"),
        "CacheBackendProtocol": ("mteb.models", "CacheBackendProtocol"),
        "CrossEncoderProtocol": ("mteb.models", "CrossEncoderProtocol"),
        "EncoderProtocol": ("mteb.models", "EncoderProtocol"),
        "IndexEncoderSearchProtocol": ("mteb.models", "IndexEncoderSearchProtocol"),
        "SearchProtocol": ("mteb.models", "SearchProtocol"),
        "SentenceTransformerEncoderWrapper": (
            "mteb.models",
            "SentenceTransformerEncoderWrapper",
        ),
        "get_model": ("mteb.models.get_model_meta", "get_model"),
        "get_model_meta": ("mteb.models.get_model_meta", "get_model_meta"),
        "get_model_metas": ("mteb.models.get_model_meta", "get_model_metas"),
        "Benchmark": ("mteb.benchmarks.benchmark", "Benchmark"),
        "get_benchmark": ("mteb.benchmarks.get_benchmark", "get_benchmark"),
        "get_benchmarks": ("mteb.benchmarks.get_benchmark", "get_benchmarks"),
    }

    if attr_name not in _module_map:
        raise AttributeError(f"module 'mteb' has no attribute '{attr_name}'")

    module_path, attr = _module_map[attr_name]

    # Import the module and get the attribute
    module = importlib.import_module(module_path)
    value = getattr(module, attr)

    # Cache for future access
    globals()[attr_name] = value
    return value


def __dir__():
    """Include lazy-loaded attributes in dir() for introspection."""
    # Get the base attributes from globals
    base = list(globals().keys())
    # Add all items from __all__ (includes lazy-loaded attrs)
    return sorted(set(base + __all__))


# Wrap the module to intercept attribute access
class _ModuleWrapper:
    """Wrapper to fix submodule shadowing issue."""

    def __init__(self, module):
        object.__setattr__(self, "_module", module)

    def __getattribute__(self, name):
        if name == "_module":
            return object.__getattribute__(self, name)

        module = object.__getattribute__(self, "_module")

        # Get the attribute from the actual module
        try:
            value = getattr(module, name)
        except AttributeError:
            raise AttributeError(f"module 'mteb' has no attribute '{name}'")

        # If this is a submodule that shadows a function/class, fix it
        if hasattr(value, "__name__") and isinstance(
            getattr(value, "__name__", None), str
        ):
            if value.__name__.startswith("mteb.") and value.__name__.count(".") == 1:
                submodule_name = value.__name__.split(".")[-1]
                if submodule_name == name and hasattr(value, name):
                    # Return the function/class from inside the module
                    actual_value = getattr(value, name)
                    # Cache it
                    setattr(module, name, actual_value)
                    return actual_value

        return value

    def __setattr__(self, name, value):
        if name == "_module":
            object.__setattr__(self, name, value)
        else:
            module = object.__getattribute__(self, "_module")
            setattr(module, name, value)

    def __delattr__(self, name):
        module = object.__getattribute__(self, "_module")
        delattr(module, name)

    def __dir__(self):
        module = object.__getattribute__(self, "_module")
        # Call the module's __dir__ function if it exists
        if hasattr(module, "__dir__") and callable(getattr(module, "__dir__", None)):
            return module.__dir__()
        return dir(module)


# Wrap the current module
_current_module = sys.modules[__name__]
sys.modules[__name__] = _ModuleWrapper(_current_module)

# Check if we're in a documentation build context
# If so, eagerly load all lazy attributes for introspection

if os.environ.get("MTEB_BUILD_DOCS") or "READTHEDOCS" in os.environ:
    # Pre-load all lazy attributes for documentation introspection
    for attr in __all__:
        if attr not in _current_module.__dict__:
            try:
                getattr(sys.modules[__name__], attr)
            except Exception:
                pass

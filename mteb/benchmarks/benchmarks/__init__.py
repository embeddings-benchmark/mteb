import importlib
from pathlib import Path

from mteb.benchmarks.benchmark import Benchmark


def _collect_benchmarks() -> tuple[list[Benchmark], list[str]]:
    """Import all Benchmark objects from all modules in the benchmarks package."""
    benchmark_objects = []
    names = []

    package_dir = Path(__file__).parent

    for file_path in sorted(package_dir.glob("*.py")):
        if file_path.name == "__init__.py":
            continue

        module = importlib.import_module(f".{file_path.stem}", package=__name__)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            if isinstance(attr, Benchmark):
                benchmark_objects.append(attr)
                names.append(attr_name)

    return benchmark_objects, names


def get_all_benchmark_objects() -> list[Benchmark]:
    """Get all Benchmark objects defined in the benchmarks package."""
    return _collect_benchmarks()[0]


# keep `from mteb.benchmarks.benchmarks import MTEB_EN` working for backward compatibility
_benchmarks, _names = _collect_benchmarks()
globals().update(zip(_names, _benchmarks, strict=True))

# build a registry of all benchmark objects
BENCHMARK_REGISTRY: dict[str, Benchmark] = {
    benchmark.name: benchmark for benchmark in _benchmarks
}

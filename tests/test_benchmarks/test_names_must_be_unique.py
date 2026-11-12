import logging

import pytest

pytestmark = pytest.mark.core

logging.basicConfig(level=logging.INFO)


def test_benchmark_names_must_be_unique():
    from mteb.benchmarks.benchmarks import get_all_benchmark_objects

    names = [inst.name for inst in get_all_benchmark_objects()]
    assert len(names) == len(set(names))


def test_all_benchmarks_exported():
    import mteb.benchmarks.benchmarks as benchmark_module

    for name, inst in benchmark_module.BENCHMARK_REGISTRY.items():
        assert any(
            getattr(benchmark_module, attr, None) is inst
            for attr in benchmark_module.__all__
        ), (
            f"Benchmark '{name}' is not exported in mteb/benchmarks/benchmarks/__init__.py"
        )

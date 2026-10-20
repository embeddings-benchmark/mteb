import logging

import pytest

pytestmark = pytest.mark.core

logging.basicConfig(level=logging.INFO)


def test_benchmark_names_must_be_unique():
    from mteb.benchmarks.benchmarks import get_all_benchmark_objects

    names = [inst.name for inst in get_all_benchmark_objects()]
    assert len(names) == len(set(names))

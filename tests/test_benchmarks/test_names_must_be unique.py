from __future__ import annotations

import logging

import mteb
import mteb.get_tasks

logging.basicConfig(level=logging.INFO)


def test_benchmark_names_must_be_unique():
    import mteb.benchmarks.benchmarks as benchmark_module

    names = [
        inst.name
        for nam, inst in benchmark_module.__dict__.items()
        if isinstance(inst, mteb.Benchmark)
    ]
    assert len(names) == len(set(names))

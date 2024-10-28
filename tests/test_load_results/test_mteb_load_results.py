from __future__ import annotations

import os
from pathlib import Path

import mteb
from mteb.load_results.benchmark_results import BenchmarkResults, ModelResult


def test_mteb_load_results():
    tests_path = Path(__file__).parent.parent / "mock_mteb_cache"

    os.environ["MTEB_CACHE"] = str(tests_path)

    results = mteb.load_results(download_latest=False)

    assert isinstance(results, BenchmarkResults)
    for model_result in results:
        assert isinstance(model_result, ModelResult)
        for res in model_result:
            assert isinstance(res, mteb.TaskResult)

    known_model = "sentence-transformers/average_word_embeddings_levy_dependency"
    known_revision = "6d9c09a789ad5dd126b476323fccfeeafcd90509"
    assert known_model in [res.model_name for res in results]
    assert known_revision in [
        res.model_revision for res in results if res.model_name == known_model
    ]

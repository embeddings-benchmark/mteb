from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from mteb import MTEB
from mteb.benchmarks import Benchmark

logging.basicConfig(level=logging.INFO)


def test_run_using_benchmark():
    """Test that a benchmark object can be run using the MTEB class."""
    model = SentenceTransformer("average_word_embeddings_komninos")
    bench = Benchmark(name="test_bench", tasks=["STS12", "SummEval"])

    eval = MTEB(tasks=bench)
    eval.run(
        model, output_folder="tests/results", overwrite_results=True
    )  # we just want to test that it runs

"""test that mteb.evaluate's integration with datasets"""

import logging
from pathlib import Path

import pytest

import mteb
from mteb.abstasks import AbsTask
from mteb.mocks import TASK_TEST_GRID

logging.basicConfig(level=logging.INFO)

DENSE_MODEL = "mteb/baseline-random-encoder"
SPARSE_MODEL = "mteb/baseline-random-sparse-encoder"
COLBERT_MODEL = "mteb/baseline-random-colbert"
CROSS_ENCODER_MODEL = "mteb/baseline-random-cross-encoder"

EXPECTED_SCORES = {
    (DENSE_MODEL, "BornholmBitextMining"): 0.00067,
    (DENSE_MODEL, "TwentyNewsgroupsClustering"): 0.04720,
    (DENSE_MODEL, "TwentyNewsgroupsClustering.v2"): 0.06383,
    (DENSE_MODEL, "LccSentimentClassification"): 0.32533,
    (DENSE_MODEL, "FarsTail"): 0.51226,
    (DENSE_MODEL, "BrazilianToxicTweetsClassification"): 0.21396,
    (DENSE_MODEL, "FaroeseSTS"): 0.00330,
    (DENSE_MODEL, "SummEval"): 0.26868,
    (DENSE_MODEL, "TwitterHjerneRetrieval"): 0.03552,
    (DENSE_MODEL, "SciDocsRR"): 0.25268,
    (DENSE_MODEL, "Core17InstructionRetrieval"): -0.01951,
    (DENSE_MODEL, "IFIRNFCorpus"): 0.0,
    (SPARSE_MODEL, "BornholmBitextMining"): 0.0,
    (SPARSE_MODEL, "TwentyNewsgroupsClustering"): 0.05676,
    (SPARSE_MODEL, "TwentyNewsgroupsClustering.v2"): 0.07125,
    (SPARSE_MODEL, "LccSentimentClassification"): 0.32600,
    (SPARSE_MODEL, "FarsTail"): 0.53990,
    (SPARSE_MODEL, "BrazilianToxicTweetsClassification"): 0.19370,
    (SPARSE_MODEL, "FaroeseSTS"): 0.02229,
    (SPARSE_MODEL, "SummEval"): 0.26734,
    (SPARSE_MODEL, "TwitterHjerneRetrieval"): 0.02910,
    (SPARSE_MODEL, "SciDocsRR"): 0.25355,
    (SPARSE_MODEL, "Core17InstructionRetrieval"): -0.01239,
    (SPARSE_MODEL, "IFIRNFCorpus"): 0.0,
    (COLBERT_MODEL, "TwitterHjerneRetrieval"): 0.01750,
    (CROSS_ENCODER_MODEL, "SciDocsRR"): 0.25268,
}


def _assert_score(result: mteb.TaskResult, model_name: str) -> None:
    expected_score = EXPECTED_SCORES[(model_name, result.task_name)]

    assert result.get_score() == pytest.approx(expected_score, abs=1e-5, nan_ok=True), (
        f"{model_name} final score changed for {result.task_name}"
    )


def _evaluate_and_assert_score(
    model: mteb.EncoderProtocol,
    task: AbsTask,
    model_name: str,
) -> None:
    task = mteb.get_task(task.metadata.name)
    result = mteb.evaluate(model, task, cache=None)[0]
    _assert_score(result, model_name)


@pytest.mark.parametrize("task", TASK_TEST_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model(DENSE_MODEL)])
def test_benchmark_datasets(task: AbsTask, model: mteb.EncoderProtocol, tmp_path: Path):
    """Test that a task can be fetched and produces the expected final score."""
    _evaluate_and_assert_score(model, task, DENSE_MODEL)


@pytest.mark.parametrize("task", TASK_TEST_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model(SPARSE_MODEL)])
def test_benchmark_datasets_sparse_encoder(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and produces the expected final score."""
    _evaluate_and_assert_score(model, task, SPARSE_MODEL)


@pytest.mark.parametrize(
    # Only one (small) task: RandomColBERTBaseline.search() scores the full corpus against all
    # queries in a single MaxSim call with no chunking, which OOMs on the larger corpora of the
    # other retrieval-shaped tasks in TASK_TEST_GRID (e.g. SciDocsRR).
    "task",
    [mteb.get_task("TwitterHjerneRetrieval")],
    ids=lambda t: t.metadata.name,
)
@pytest.mark.parametrize("model", [mteb.get_model(COLBERT_MODEL)])
def test_benchmark_datasets_colbert(task: AbsTask, model: mteb.EncoderProtocol):
    _evaluate_and_assert_score(model, task, COLBERT_MODEL)


@pytest.mark.parametrize(
    "task",
    [t for t in TASK_TEST_GRID if t.metadata.type == "Reranking"],
    ids=lambda t: t.metadata.name,
)
@pytest.mark.parametrize("model", [mteb.get_model(CROSS_ENCODER_MODEL)])
def test_benchmark_datasets_cross_encoder(task: AbsTask, model: mteb.EncoderProtocol):
    _evaluate_and_assert_score(model, task, CROSS_ENCODER_MODEL)


def test_run_task_multiple_times():
    """Regression test for https://github.com/embeddings-benchmark/mteb/issues/4397"""
    model = mteb.get_model(DENSE_MODEL)
    # Core17InstructionRetrieval already in TASK_TEST_GRID
    task = mteb.get_task("Core17InstructionRetrieval")
    first_result = mteb.evaluate(model, task, cache=None)[0]
    second_result = mteb.evaluate(model, task, cache=None)[0]
    _assert_score(first_result, DENSE_MODEL)
    _assert_score(second_result, DENSE_MODEL)

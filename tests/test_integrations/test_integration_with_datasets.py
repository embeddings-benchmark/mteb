"""test that mteb.evaluate's integration with datasets"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import sklearn
from packaging.version import Version

import mteb
from mteb.abstasks import AbsTask
from mteb.mocks import TASK_TEST_GRID

logging.basicConfig(level=logging.INFO)


@dataclass
class ModelInfo:
    name: str
    expected_scores: dict[str, float]


DENSE_MODEL = ModelInfo(
    name="mteb/baseline-random-encoder",
    expected_scores={
        "BornholmBitextMining": 0.00067,
        "TwentyNewsgroupsClustering": 0.04720,
        "TwentyNewsgroupsClustering.v2": 0.06383,
        "LccSentimentClassification": 0.32533,
        "FarsTail": 0.51226,
        "BrazilianToxicTweetsClassification": 0.21396,
        "FaroeseSTS": 0.00330,
        "SummEval": 0.26868,
        "TwitterHjerneRetrieval": 0.03552,
        "SciDocsRR": 0.25268,
        "Core17InstructionRetrieval": -0.01951,
        "IFIRNFCorpus": 0.0,
    },
)
SPARSE_MODEL = ModelInfo(
    name="mteb/baseline-random-sparse-encoder",
    expected_scores={
        "BornholmBitextMining": 0.0,
        "TwentyNewsgroupsClustering": 0.05676,
        "TwentyNewsgroupsClustering.v2": 0.07125,
        "LccSentimentClassification": 0.32600,
        "FarsTail": 0.53992,
        "BrazilianToxicTweetsClassification": 0.19370,
        "FaroeseSTS": 0.02216,
        "SummEval": 0.26734,
        "TwitterHjerneRetrieval": 0.02910,
        "SciDocsRR": 0.25355,
        "Core17InstructionRetrieval": -0.01239,
        "IFIRNFCorpus": 0.0,
    },
)
COLBERT_MODEL = ModelInfo(
    name="mteb/baseline-random-colbert",
    expected_scores={"TwitterHjerneRetrieval": 0.01750},
)
CROSS_ENCODER_MODEL = ModelInfo(
    name="mteb/baseline-random-cross-encoder",
    expected_scores={"SciDocsRR": 0.25268},
)

# The minimum supported scikit-learn releases produce different deterministic KMeans
# assignments than 1.8+ for the same fixed embeddings and random state.
if Version(sklearn.__version__) < Version("1.8.0"):
    DENSE_MODEL.expected_scores.update(
        {
            "TwentyNewsgroupsClustering": 0.04560,
            "TwentyNewsgroupsClustering.v2": 0.06337,
        }
    )
    SPARSE_MODEL.expected_scores.update(
        {
            "TwentyNewsgroupsClustering": 0.05358,
            "TwentyNewsgroupsClustering.v2": 0.07122,
        }
    )

# The macOS numerical stack produces slightly different sparse pair-classification and STS
# scores than the Linux/Windows stacks. Keep strict expected scores for each platform family.
if sys.platform == "darwin":
    SPARSE_MODEL.expected_scores.update(
        {
            "FarsTail": 0.53990,
            "FaroeseSTS": 0.02229,
        }
    )


def _assert_score(result: mteb.TaskResult, model_info: ModelInfo) -> None:
    expected_score = model_info.expected_scores[result.task_name]

    assert result.get_score() == pytest.approx(expected_score, abs=1e-5, nan_ok=True), (
        f"{model_info.name} final score changed for {result.task_name}"
    )


def _evaluate_and_assert_score(
    model: mteb.EncoderProtocol,
    task: AbsTask,
    model_info: ModelInfo,
) -> None:
    task = mteb.get_task(task.metadata.name)
    result = mteb.evaluate(model, task, cache=None)[0]
    _assert_score(result, model_info)


@pytest.mark.parametrize("task", TASK_TEST_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model(DENSE_MODEL.name)])
def test_benchmark_datasets(task: AbsTask, model: mteb.EncoderProtocol, tmp_path: Path):
    """Test that a task can be fetched and produces the expected final score."""
    _evaluate_and_assert_score(model, task, DENSE_MODEL)


@pytest.mark.parametrize("task", TASK_TEST_GRID, ids=lambda t: t.metadata.name)
@pytest.mark.parametrize("model", [mteb.get_model(SPARSE_MODEL.name)])
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
@pytest.mark.parametrize("model", [mteb.get_model(COLBERT_MODEL.name)])
def test_benchmark_datasets_colbert(task: AbsTask, model: mteb.EncoderProtocol):
    _evaluate_and_assert_score(model, task, COLBERT_MODEL)


@pytest.mark.parametrize(
    "task",
    [t for t in TASK_TEST_GRID if t.metadata.type == "Reranking"],
    ids=lambda t: t.metadata.name,
)
@pytest.mark.parametrize("model", [mteb.get_model(CROSS_ENCODER_MODEL.name)])
def test_benchmark_datasets_cross_encoder(task: AbsTask, model: mteb.EncoderProtocol):
    _evaluate_and_assert_score(model, task, CROSS_ENCODER_MODEL)


def test_run_task_multiple_times():
    """Regression test for https://github.com/embeddings-benchmark/mteb/issues/4397"""
    model = mteb.get_model(DENSE_MODEL.name)
    # Core17InstructionRetrieval already in TASK_TEST_GRID
    task = mteb.get_task("Core17InstructionRetrieval")
    first_result = mteb.evaluate(model, task, cache=None)[0]
    second_result = mteb.evaluate(model, task, cache=None)[0]
    _assert_score(first_result, DENSE_MODEL)
    _assert_score(second_result, DENSE_MODEL)

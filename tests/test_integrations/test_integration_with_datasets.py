"""test that mteb.evaluate's integration with datasets"""

import logging
import sys

import pytest
import sklearn
from packaging.version import Version

import mteb
from mteb.abstasks import AbsTask
from tests.test_integrations._model_info import ModelInfo, assert_final_score

logging.basicConfig(level=logging.INFO)


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


@pytest.mark.parametrize(("model", "task", "expected_score"), DENSE_MODEL)
def test_benchmark_datasets(
    model: mteb.EncoderProtocol, task: AbsTask, expected_score: float
):
    """Test that a task can be fetched and produces the expected final score."""
    assert_final_score(model, task, expected_score)


@pytest.mark.parametrize(("model", "task", "expected_score"), SPARSE_MODEL)
def test_benchmark_datasets_sparse_encoder(
    model: mteb.EncoderProtocol, task: AbsTask, expected_score: float
):
    """Test that a task can be fetched and produces the expected final score."""
    assert_final_score(model, task, expected_score)


@pytest.mark.parametrize(("model", "task", "expected_score"), COLBERT_MODEL)
def test_benchmark_datasets_colbert(
    model: mteb.EncoderProtocol, task: AbsTask, expected_score: float
):
    assert_final_score(model, task, expected_score)


@pytest.mark.parametrize(("model", "task", "expected_score"), CROSS_ENCODER_MODEL)
def test_benchmark_datasets_cross_encoder(
    model: mteb.EncoderProtocol, task: AbsTask, expected_score: float
):
    assert_final_score(model, task, expected_score)


def test_run_task_multiple_times():
    """Regression test for https://github.com/embeddings-benchmark/mteb/issues/4397"""
    task = mteb.get_task("Core17InstructionRetrieval")
    expected_score = DENSE_MODEL.expected_scores[task.metadata.name]

    first_result = mteb.evaluate(DENSE_MODEL.model, task, cache=None)[0]
    second_result = mteb.evaluate(DENSE_MODEL.model, task, cache=None)[0]

    for result in (first_result, second_result):
        assert result.get_score() == pytest.approx(
            expected_score, abs=1e-5, nan_ok=True
        )

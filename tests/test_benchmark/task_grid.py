"""Task grid for testing purposes. This is a list of tasks that can be used to test the benchmarking pipeline."""

from __future__ import annotations

import mteb
from mteb.abstasks import AbsTask

from .mock_tasks import (
    MockAny2AnyRetrievalI2TTask,
    MockAny2AnyRetrievalT2ITask,
    MockBitextMiningTask,
    MockClassificationTask,
    MockClusteringFastTask,
    MockClusteringTask,
    MockImageClassificationKNNPTTask,
    MockImageClassificationKNNTask,
    MockImageClassificationTask,
    MockImageClusteringTask,
    MockImageMultilabelClassificationTask,
    MockImageTextPairClassificationTask,
    MockInstructionReranking,
    MockInstructionRetrieval,
    MockMultiChoiceTask,
    MockMultilabelClassification,
    MockMultilingualBitextMiningTask,
    MockMultilingualClassificationTask,
    MockMultilingualClusteringFastTask,
    MockMultilingualClusteringTask,
    MockMultilingualImageClassificationTask,
    MockMultilingualImageMultilabelClassificationTask,
    MockMultilingualImageTextPairClassificationTask,
    MockMultilingualInstructionReranking,
    MockMultilingualInstructionRetrieval,
    MockMultilingualMultiChoiceTask,
    MockMultilingualMultilabelClassification,
    MockMultilingualPairClassificationTask,
    MockMultilingualParallelBitextMiningTask,
    MockMultilingualRerankingTask,
    MockMultilingualRetrievalTask,
    MockMultilingualSTSTask,
    MockMultilingualSummarizationTask,
    MockPairClassificationTask,
    MockRerankingTask,
    MockRetrievalTask,
    MockSTSTask,
    MockSummarizationTask,
    MockTextMultipleChoiceTask,
    MockVisualSTSTask,
    MockZeroshotClassificationTask,
)

TASK_TEST_GRID = (
    mteb.get_tasks(
        tasks=[
            "BornholmBitextMining",  # bitext mining + just supplying a task class instead of a string
            "TwentyNewsgroupsClustering",  # clustering and string instead of class
            "TwentyNewsgroupsClustering.v2",  # fast clustering
            "Banking77Classification",  # classification
            "FarsTail",  # pair classification
            "BrazilianToxicTweetsClassification",  # multilabel classification
            "FaroeseSTS",  # STS
            "SummEval",  # summarization
            "TwitterHjerneRetrieval",  # retrieval
            "SciDocsRR",  # reranking
            "Core17InstructionRetrieval",  # instruction reranking
            "InstructIR",  # instruction retrieval
        ]
    )
    + mteb.get_tasks(tasks=["IndicSentimentClassification"], languages=["asm-Beng"])
)

TASK_TEST_GRID_AS_STRING = [
    t.metadata.name if isinstance(t, AbsTask) else t for t in TASK_TEST_GRID
]


# Mock tasks for testing - intended to be faster and avoid downloading data leading to false positive potential failures in CI
# Not all tasks are implemented as Mock tasks yet
MOCK_TASK_TEST_GRID = [
    MockBitextMiningTask(),
    MockMultilingualBitextMiningTask(),
    MockMultilingualParallelBitextMiningTask(),
    MockClassificationTask(),
    MockMultilingualClassificationTask(),
    MockClusteringTask(),
    MockMultilingualClusteringTask(),
    MockClusteringFastTask(),
    MockMultilingualClusteringFastTask(),
    MockPairClassificationTask(),
    MockMultilingualPairClassificationTask(),
    MockRerankingTask(),
    MockMultilingualRerankingTask(),
    MockRetrievalTask(),
    MockMultilingualRetrievalTask(),
    MockSTSTask(),
    MockMultilingualSTSTask(),
    MockMultilabelClassification(),
    MockMultilingualMultilabelClassification(),
    MockSummarizationTask(),
    MockMultilingualSummarizationTask(),
    MockInstructionRetrieval(),
    MockMultilingualInstructionRetrieval(),
    MockMultilingualInstructionReranking(),
    MockInstructionReranking(),
]

MOCK_TASK_TEST_GRID_AS_STRING = [
    t.metadata.name if isinstance(t, AbsTask) else t for t in MOCK_TASK_TEST_GRID
]

MOCK_TASK_REGISTRY = {task.metadata.name: type(task) for task in MOCK_TASK_TEST_GRID}

MOCK_MIEB_TASK_GRID = [
    MockAny2AnyRetrievalI2TTask(),
    MockAny2AnyRetrievalT2ITask(),
    MockTextMultipleChoiceTask(),
    MockMultiChoiceTask(),
    MockImageClassificationTask(),
    MockImageClassificationKNNPTTask(),
    MockImageClassificationKNNTask(),
    MockImageClusteringTask(),
    MockImageTextPairClassificationTask(),
    MockVisualSTSTask(),
    MockZeroshotClassificationTask(),
    MockImageMultilabelClassificationTask(),
    MockMultilingualImageClassificationTask(),
    MockMultilingualImageTextPairClassificationTask(),
    MockMultilingualMultiChoiceTask(),
    MockMultilingualImageMultilabelClassificationTask(),
]

MOCK_MIEB_TASK_GRID_AS_STRING = [
    t.metadata.name if isinstance(t, AbsTask) else t for t in MOCK_MIEB_TASK_GRID
]

MOCK_MIEB_TASK_REGISTRY = {
    task.metadata.name: type(task) for task in MOCK_MIEB_TASK_GRID
}

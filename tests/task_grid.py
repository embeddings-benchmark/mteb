"""Task grid for testing purposes. This is a list of tasks that can be used to test the benchmarking pipeline."""

import mteb
from mteb.abstasks import AbsTask

from .mock_tasks import (
    MockAny2AnyRetrievalI2TTask,
    MockAny2AnyRetrievalT2ITask,
    MockBitextMiningTask,
    MockClassificationTask,
    MockClusteringFastTask,
    MockClusteringTask,
    MockImageClassificationTask,
    MockImageClusteringFastTask,
    MockImageClusteringTask,
    MockImageMultilabelClassificationTask,
    MockImageRegressionTask,
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
    MockPairImageClassificationTask,
    MockRegressionTask,
    MockRerankingTask,
    MockRetrievalDialogTask,
    MockRetrievalTask,
    MockSTSTask,
    MockSummarizationTask,
    MockTextZeroShotClassificationTask,
    MockVisualSTSTask,
    MockZeroShotClassificationTask,
)

TASK_TEST_GRID = mteb.get_tasks(
    tasks=[
        "BornholmBitextMining",  # bitext mining + just supplying a task class instead of a string
        "TwentyNewsgroupsClustering",  # clustering and string instead of class
        "TwentyNewsgroupsClustering.v2",  # fast clustering
        "LccSentimentClassification",  # classification
        "FarsTail",  # pair classification
        "BrazilianToxicTweetsClassification",  # multilabel classification
        "FaroeseSTS",  # STS
        "SummEval",  # summarization
        "TwitterHjerneRetrieval",  # retrieval
        "SciDocsRR",  # reranking
        "Core17InstructionRetrieval",  # instruction reranking
        "IFIRNFCorpus",  # instruction retrieval
    ]
)

TASK_TEST_GRID_AS_STRING = [
    t.metadata.name if isinstance(t, AbsTask) else t for t in TASK_TEST_GRID
]


# Mock tasks for testing - intended to be faster and avoid downloading data leading to false positive potential failures in CI
# Not all tasks are implemented as Mock tasks yet
MOCK_TASK_TEST_GRID_MULTILINGUAL = [
    MockMultilingualBitextMiningTask(),
    MockMultilingualParallelBitextMiningTask(),
    MockMultilingualClassificationTask(),
    MockMultilingualClusteringTask(),
    MockMultilingualClusteringFastTask(),
    MockMultilingualPairClassificationTask(),
    MockMultilingualRerankingTask(),
    MockMultilingualRetrievalTask(),
    MockMultilingualSTSTask(),
    MockMultilingualMultilabelClassification(),
    MockMultilingualSummarizationTask(),
    MockMultilingualInstructionRetrieval(),
    MockMultilingualInstructionReranking(),
]

MOCK_TASK_TEST_GRID_MONOLINGUAL = [
    MockBitextMiningTask(),
    MockClassificationTask(),
    MockRegressionTask(),
    MockClusteringTask(),
    MockClusteringFastTask(),
    MockPairClassificationTask(),
    MockRerankingTask(),
    MockRetrievalTask(),
    MockSTSTask(),
    MockMultilingualSTSTask(),
    MockMultilabelClassification(),
    MockSummarizationTask(),
    MockInstructionRetrieval(),
    MockInstructionReranking(),
    MockRetrievalDialogTask(),
    MockTextZeroShotClassificationTask(),
]


MOCK_TASK_TEST_GRID = MOCK_TASK_TEST_GRID_MULTILINGUAL + MOCK_TASK_TEST_GRID_MONOLINGUAL

MOCK_TASK_TEST_GRID_AS_STRING = [
    t.metadata.name if isinstance(t, AbsTask) else t for t in MOCK_TASK_TEST_GRID
]

MOCK_TASK_REGISTRY = {task.metadata.name: type(task) for task in MOCK_TASK_TEST_GRID}

MOCK_MIEB_TASK_GRID = [
    MockAny2AnyRetrievalI2TTask(),
    MockAny2AnyRetrievalT2ITask(),
    MockMultiChoiceTask(),
    MockImageClassificationTask(),
    MockImageClusteringTask(),
    MockImageTextPairClassificationTask(),
    MockVisualSTSTask(),
    MockZeroShotClassificationTask(),
    MockImageMultilabelClassificationTask(),
    MockMultilingualImageClassificationTask(),
    MockMultilingualImageTextPairClassificationTask(),
    MockMultilingualMultiChoiceTask(),
    MockMultilingualImageMultilabelClassificationTask(),
    MockImageClusteringFastTask(),
    MockImageRegressionTask(),
    MockPairImageClassificationTask(),
]

MOCK_MIEB_TASK_GRID_AS_STRING = [
    t.metadata.name if isinstance(t, AbsTask) else t for t in MOCK_MIEB_TASK_GRID
]

MOCK_MIEB_TASK_REGISTRY = {
    task.metadata.name: type(task) for task in MOCK_MIEB_TASK_GRID
}

ALL_TASK_TEST_GRID = MOCK_TASK_TEST_GRID + MOCK_MIEB_TASK_GRID

ALL_TASK_TEST_GRID_AS_STRING = [
    t.metadata.name if isinstance(t, AbsTask) else t for t in ALL_TASK_TEST_GRID
]

ALL_MOCK_TASK_REGISTRY = {task.metadata.name: type(task) for task in ALL_TASK_TEST_GRID}

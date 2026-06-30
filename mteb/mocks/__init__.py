"""Mocks package entrypoint. This defines and exposes various task grids and mock task components."""

from __future__ import annotations

import mteb
from mteb.abstasks import AbsTask

from .mock_tasks import (
    LegacyMockClusteringFastTask,
    MockAny2AnyRetrievalA2ATask,
    MockAny2AnyRetrievalA2TTask,
    MockAny2AnyRetrievalI2TTask,
    MockAny2AnyRetrievalT2ATask,
    MockAny2AnyRetrievalT2ITask,
    MockAsymVideoAudioPairClassificationTask,
    MockAsymVideoAudioPairClassificationTaskV2,
    MockAudioClassification,
    MockAudioClassificationCrossVal,
    MockAudioClusteringTask,
    MockAudioMultilabelClassificationTask,
    MockAudioPairClassification,
    MockAudioReranking,
    MockAudioZeroshotClassificationTask,
    MockBitextMiningTask,
    MockClassificationTask,
    MockImageClassificationTask,
    MockImageClusteringFastTask,
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
    MockMultilingualImageClassificationTask,
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
    MockSymCustomVideoAudioPairClassificationTaskV2,
    MockSymCustomVideoAudiSTSTask,
    MockTextZeroShotClassificationTask,
    MockVideoAudioClassification,
    MockVideoAudioClusteringTask,
    MockVideoAudioMultilabelClassificationTask,
    MockVideoAudioPairClassificationTask,
    MockVideoAudioRetrievalT2VA,
    MockVideoAudioRetrievalVA2T,
    MockVideoAudioSTSTask,
    MockVideoAudioTextRetrievalVAT2T,
    MockVideoAudioZeroshotClassificationTask,
    MockVideoClassification,
    MockVideoClusteringTask,
    MockVideoMultilabelClassificationTask,
    MockVideoPairClassificationTask,
    MockVideoRetrievalT2V,
    MockVideoRetrievalV2T,
    MockVideoZeroshotClassificationTask,
    MockVisualSTSTask,
    MockZeroShotClassificationTask,
)
from .mock_tasks.clustering import (
    MockClusteringTask,
    MockImageClusteringTask,
    MockMultilingualClusteringTask,
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
    LegacyMockClusteringFastTask(),
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
    MockImageClusteringFastTask(),
    MockImageRegressionTask(),
    MockPairImageClassificationTask(),
]

MOCK_MAEB_TASK_GRID = [
    MockAudioClusteringTask(),
    MockAudioMultilabelClassificationTask(),
    MockAudioZeroshotClassificationTask(),
    MockAny2AnyRetrievalT2ATask(),
    MockAny2AnyRetrievalA2TTask(),
    MockAny2AnyRetrievalA2ATask(),
    MockAudioReranking(),
    MockAudioClassification(),
    MockAudioClassificationCrossVal(),
    MockAudioPairClassification(),
]

MOCK_MIEB_TASK_GRID_AS_STRING = [
    t.metadata.name if isinstance(t, AbsTask) else t for t in MOCK_MIEB_TASK_GRID
]

MOCK_MIEB_TASK_REGISTRY = {
    task.metadata.name: type(task) for task in MOCK_MIEB_TASK_GRID
}

MOCK_MVEB_TASK_GRID = [
    MockVideoClassification(),
    MockVideoClusteringTask(),
    MockVideoMultilabelClassificationTask(),
    MockVideoZeroshotClassificationTask(),
    MockVideoPairClassificationTask(),
    MockVideoRetrievalV2T(),
    MockVideoRetrievalT2V(),
]

MOCK_MULTIMODAL_TASKS = (
    MockVideoAudioClassification(),
    MockVideoAudioClusteringTask(),
    MockVideoAudioMultilabelClassificationTask(),
    MockVideoAudioPairClassificationTask(),
    MockVideoAudioSTSTask(),
    MockVideoAudioRetrievalVA2T(),
    MockVideoAudioRetrievalT2VA(),
    MockVideoAudioTextRetrievalVAT2T(),
    MockVideoAudioZeroshotClassificationTask(),
    MockAsymVideoAudioPairClassificationTask(),
    MockAsymVideoAudioPairClassificationTaskV2(),
    MockSymCustomVideoAudioPairClassificationTaskV2(),
    MockSymCustomVideoAudiSTSTask(),
)

ALL_TASK_TEST_GRID = (
    MOCK_TASK_TEST_GRID
    + MOCK_MIEB_TASK_GRID
    + MOCK_MAEB_TASK_GRID
    + MOCK_MVEB_TASK_GRID
)

ALL_TASK_TEST_GRID_AS_STRING = [
    t.metadata.name if isinstance(t, AbsTask) else t for t in ALL_TASK_TEST_GRID
]

ALL_MOCK_TASK_REGISTRY = {task.metadata.name: type(task) for task in ALL_TASK_TEST_GRID}

task_grid = TASK_TEST_GRID

__all__ = [
    "ALL_MOCK_TASK_REGISTRY",
    "ALL_TASK_TEST_GRID",
    "ALL_TASK_TEST_GRID_AS_STRING",
    "MOCK_MAEB_TASK_GRID",
    "MOCK_MIEB_TASK_GRID",
    "MOCK_MIEB_TASK_GRID_AS_STRING",
    "MOCK_MIEB_TASK_REGISTRY",
    "MOCK_MULTIMODAL_TASKS",
    "MOCK_MVEB_TASK_GRID",
    "MOCK_TASK_REGISTRY",
    "MOCK_TASK_TEST_GRID",
    "MOCK_TASK_TEST_GRID_AS_STRING",
    "TASK_TEST_GRID",
    "TASK_TEST_GRID_AS_STRING",
    "task_grid",
]

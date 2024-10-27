"""Task grid for testing purposes. This is a list of tasks that can be used to test the benchmarking pipeline."""

from __future__ import annotations

from mteb.abstasks import AbsTask
from mteb.tasks.BitextMining.dan.BornholmskBitextMining import BornholmBitextMining
from mteb.tasks.Classification.multilingual.IndicSentimentClassification import (
    IndicSentimentClassification,
)
from mteb.tasks.Clustering.eng.TwentyNewsgroupsClustering import (
    TwentyNewsgroupsClusteringFast,
)
from mteb.tasks.Image.Clustering import TinyImageNet
from mteb.tasks.Image.ImageTextPairClassification import Winoground
from mteb.tasks.Image.ZeroshotClassification import RenderedSST2

from .mock_tasks import (
    MockBitextMiningTask,
    MockClassificationTask,
    MockClusteringFastTask,
    MockClusteringTask,
    MockInstructionRetrival,
    MockMultilabelClassification,
    MockMultilingualBitextMiningTask,
    MockMultilingualClassificationTask,
    MockMultilingualClusteringFastTask,
    MockMultilingualClusteringTask,
    MockMultilingualInstructionRetrival,
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
)

twenty_news = TwentyNewsgroupsClusteringFast()

# downsample to speed up tests
twenty_news.max_document_to_embed = 1000
twenty_news.n_clusters = 2
twenty_news.max_fraction_of_documents_to_embed = None

TASK_TEST_GRID = [
    BornholmBitextMining(),  # bitext mining + just supplying a task class instead of a string
    IndicSentimentClassification(  # multi subset loader
        hf_subsets=["as"],  # we only load one subset here to speed up tests
        n_experiments=2,  # to speed up the test
    ),
    "TwentyNewsgroupsClustering",  # clustering and string instead of class
    twenty_news,  # fast clustering
    "Banking77Classification",  # classification
    "SciDocsRR",  # reranking
    "FarsTail",  # pair classification
    "TwitterHjerneRetrieval",  # retrieval
    "BrazilianToxicTweetsClassification",  # multilabel classification
    "FaroeseSTS",  # STS
    "SummEval",  # summarization
]

TASK_TEST_GRID_AS_STRING = [
    t.metadata.name if isinstance(t, AbsTask) else t for t in TASK_TEST_GRID
]


def dataset_transform(self):
    for split in self.metadata.eval_splits:
        self.dataset[split] = self.dataset[split].select([0, 1])


tiny_imagenet = TinyImageNet()
renderedSST2 = RenderedSST2()
winoground = Winoground()

## method override to speed up tests
tiny_imagenet.dataset_transform = dataset_transform.__get__(tiny_imagenet)
renderedSST2.dataset_transform = dataset_transform.__get__(renderedSST2)
winoground.dataset_transform = dataset_transform.__get__(winoground)


MIEB_TASK_TEST_GRID = [
    tiny_imagenet,  # image clustering
    # winoground,  # pair classification. Gated
    renderedSST2,  # zero shot classification
    # The following takes a long time. Consider creating a mock class.
    # "CIRRIT2TRetrieval",  # it2i retrieval
    # "MSCOCOI2TRetrieval",  # i2t retrieval
    # "MSCOCOT2IRetrieval",  # t2i retrieval
    # oxford_flowers,  # image classification
]

MIEB_TASK_TEST_GRID_AS_STRING = [
    t.metadata.name if isinstance(t, AbsTask) else t for t in MIEB_TASK_TEST_GRID
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
    MockInstructionRetrival(),
    MockMultilingualInstructionRetrival(),
]

MOCK_TASK_TEST_GRID_AS_STRING = [
    t.metadata.name if isinstance(t, AbsTask) else t for t in MOCK_TASK_TEST_GRID
]

MOCK_TASK_REGISTRY = {task.metadata.name: type(task) for task in MOCK_TASK_TEST_GRID}

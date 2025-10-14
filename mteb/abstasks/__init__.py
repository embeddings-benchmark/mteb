from .abstask import AbsTask
from .any_classification import AbsTaskAnyClassification
from .any_sts import AbsTaskAnySTS
from .clustering import AbsTaskClustering
from .clustering_legacy import AbsTaskClusteringLegacy
from .image.image_text_pair_classification import AbsTaskImageTextPairClassification
from .multilabel_classification import AbsTaskMultilabelClassification
from .regression import AbsTaskRegression
from .retrieval import AbsTaskRetrieval
from .text.bitext_mining import AbsTaskBitextMining
from .text.pair_classification import AbsTaskPairClassification
from .text.reranking import AbsTaskReranking
from .text.summarization import AbsTaskSummarization
from .zeroshot_classification import AbsTaskZeroShotClassification

__all__ = [
    "AbsTask",
    "AbsTaskAnyClassification",
    "AbsTaskAnySTS",
    "AbsTaskBitextMining",
    "AbsTaskClustering",
    "AbsTaskClusteringLegacy",
    "AbsTaskImageTextPairClassification",
    "AbsTaskMultilabelClassification",
    "AbsTaskPairClassification",
    "AbsTaskRegression",
    "AbsTaskReranking",
    "AbsTaskRetrieval",
    "AbsTaskSummarization",
    "AbsTaskZeroShotClassification",
]

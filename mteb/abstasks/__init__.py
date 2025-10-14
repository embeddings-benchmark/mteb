from .abstask import AbsTask
from .any_classification import AbsTaskAnyClassification
from .any_clustering import AbsTaskClusteringLegacy
from .any_sts import AbsTaskAnySTS
from .any_zeroshot_classification import AbsTaskAnyZeroShotClassification
from .clustering_fast import AbsTaskClustering
from .image.image_text_pair_classification import AbsTaskImageTextPairClassification
from .multilabel_classification import AbsTaskMultilabelClassification
from .regression import AbsTaskRegression
from .retrieval import AbsTaskRetrieval
from .text.bitext_mining import AbsTaskBitextMining
from .text.pair_classification import AbsTaskPairClassification
from .text.reranking import AbsTaskReranking
from .text.summarization import AbsTaskSummarization

__all__ = [
    "AbsTask",
    "AbsTaskAnyClassification",
    "AbsTaskAnySTS",
    "AbsTaskAnyZeroShotClassification",
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
]

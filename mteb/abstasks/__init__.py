from .abstask import AbsTask
from .any_classification import AbsTaskAnyClassification
from .any_clustering import AbsTaskAnyClustering
from .any_sts import AbsTaskAnySTS
from .any_zeroshot_classification import AbsTaskAnyZeroShotClassification
from .image.image_text_pair_classification import AbsTaskImageTextPairClassification
from .text.bitext_mining import AbsTaskBitextMining
from .text.clustering_fast import AbsTaskClusteringFast
from .text.multilabel_classification import AbsTaskMultilabelClassification
from .text.pair_classification import AbsTaskPairClassification
from .text.reranking import AbsTaskReranking
from .text.retrieval import AbsTaskRetrieval
from .text.summarization import AbsTaskSummarization
from .text.text_regression import AbsTaskTextRegression

__all__ = [
    "AbsTask",
    "AbsTaskAnyClassification",
    "AbsTaskAnyClustering",
    "AbsTaskAnySTS",
    "AbsTaskAnyZeroShotClassification",
    "AbsTaskBitextMining",
    "AbsTaskClusteringFast",
    "AbsTaskImageTextPairClassification",
    "AbsTaskMultilabelClassification",
    "AbsTaskPairClassification",
    "AbsTaskReranking",
    "AbsTaskRetrieval",
    "AbsTaskSummarization",
    "AbsTaskTextRegression",
]

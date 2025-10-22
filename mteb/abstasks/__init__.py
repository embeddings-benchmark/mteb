from __future__ import annotations

from .abstask import AbsTask
from .audio.abs_task_audio_classification import AbsTaskAudioClassification
from .audio.abs_task_audio_clustering import AbsTaskAudioClustering
from .audio.abs_task_audio_pair_classification import AbsTaskAudioPairClassification
from .classification import AbsTaskClassification
from .clustering import AbsTaskClustering
from .clustering_legacy import AbsTaskClusteringLegacy
from .image.image_text_pair_classification import AbsTaskImageTextPairClassification
from .multilabel_classification import AbsTaskMultilabelClassification
from .pair_classification import AbsTaskPairClassification
from .regression import AbsTaskRegression
from .retrieval import AbsTaskRetrieval
from .sts import AbsTaskSTS
from .text.bitext_mining import AbsTaskBitextMining
from .text.reranking import AbsTaskReranking
from .text.summarization import AbsTaskSummarization
from .zeroshot_classification import AbsTaskZeroShotClassification

__all__ = [
    "AbsTask",
    "AbsTaskAudioClassification",
    "AbsTaskAudioClustering",
    "AbsTaskAudioPairClassification",
    "AbsTaskBitextMining",
    "AbsTaskClassification",
    "AbsTaskClustering",
    "AbsTaskClusteringLegacy",
    "AbsTaskImageTextPairClassification",
    "AbsTaskMultilabelClassification",
    "AbsTaskPairClassification",
    "AbsTaskRegression",
    "AbsTaskReranking",
    "AbsTaskRetrieval",
    "AbsTaskSTS",
    "AbsTaskSummarization",
    "AbsTaskZeroShotClassification",
]

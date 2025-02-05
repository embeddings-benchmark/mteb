from __future__ import annotations

from .BitextMiningEvaluator import BitextMiningEvaluator
from .ClassificationEvaluator import (
    dot_distance,
    kNNClassificationEvaluator,
    kNNClassificationEvaluatorPytorch,
    logRegClassificationEvaluator,
)
from .ClusteringEvaluator import ClusteringEvaluator
from .Evaluator import Evaluator

# TODO TODO
from .Image.Any2AnyMultiChoiceEvaluator import *
from .Image.Any2AnyRetrievalEvaluator import *
from .Image.Any2TextMultipleChoiceEvaluator import *
from .Image.ClassificationEvaluator import *
from .Image.ClusteringEvaluator import *
from .Image.ImageTextPairClassificationEvaluator import *
from .Image.VisualSTSEvaluator import *
from .Image.ZeroshotClassificationEvaluator import *
from .model_classes import DenseRetrievalExactSearch, corpus_to_str
from .PairClassificationEvaluator import PairClassificationEvaluator
from .RetrievalEvaluator import RetrievalEvaluator
from .STSEvaluator import STSEvaluator
from .SummarizationEvaluator import (
    DeprecatedSummarizationEvaluator,
    SummarizationEvaluator,
)

__all__ = [
    "Evaluator",
    "STSEvaluator",
    "SummarizationEvaluator",
    "DeprecatedSummarizationEvaluator",
    "RetrievalEvaluator",
    "DenseRetrievalExactSearch",
    "ClusteringEvaluator",
    "BitextMiningEvaluator",
    "PairClassificationEvaluator",
    "corpus_to_str",
    "kNNClassificationEvaluator",
    "kNNClassificationEvaluatorPytorch",
    "logRegClassificationEvaluator",
    "dot_distance",
]

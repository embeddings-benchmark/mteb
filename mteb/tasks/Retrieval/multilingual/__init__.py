from __future__ import annotations

from .BelebeleRetrieval import BelebeleRetrieval
from .CrossLingualSemanticDiscriminationWMT19 import (
    CrossLingualSemanticDiscriminationWMT19,
)
from .CrossLingualSemanticDiscriminationWMT21 import (
    CrossLingualSemanticDiscriminationWMT21,
)
from .CUREv1Retrieval import CUREv1Retrieval
from .IndicQARetrieval import IndicQARetrieval
from .MintakaRetrieval import MintakaRetrieval
from .MIRACLRetrieval import MIRACLRetrieval, MIRACLRetrievalHardNegatives
from .MLQARetrieval import MLQARetrieval
from .MrTidyRetrieval import MrTidyRetrieval
from .MultiLongDocRetrieval import MultiLongDocRetrieval
from .NeuCLIR2022Retrieval import (
    NeuCLIR2022Retrieval,
    NeuCLIR2022RetrievalHardNegatives,
)
from .NeuCLIR2023Retrieval import (
    NeuCLIR2023Retrieval,
    NeuCLIR2023RetrievalHardNegatives,
)
from .PublicHealthQARetrieval import PublicHealthQARetrieval
from .StatcanDialogueDatasetRetrieval import StatcanDialogueDatasetRetrieval
from .WikipediaRetrievalMultilingual import WikipediaRetrievalMultilingual
from .XMarketRetrieval import XMarket
from .XPQARetrieval import XPQARetrieval
from .XQuADRetrieval import XQuADRetrieval

__all__ = [
    "WikipediaRetrievalMultilingual",
    "MintakaRetrieval",
    "PublicHealthQARetrieval",
    "CrossLingualSemanticDiscriminationWMT19",
    "MultiLongDocRetrieval",
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "NeuCLIR2022Retrieval",
    "NeuCLIR2022RetrievalHardNegatives",
    "StatcanDialogueDatasetRetrieval",
    "IndicQARetrieval",
    "NeuCLIR2023Retrieval",
    "NeuCLIR2023RetrievalHardNegatives",
    "CrossLingualSemanticDiscriminationWMT21",
    "XMarket",
    "XPQARetrieval",
    "BelebeleRetrieval",
    "CUREv1Retrieval",
    "MLQARetrieval",
    "XQuADRetrieval",
    "MrTidyRetrieval",
]

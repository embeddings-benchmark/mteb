from __future__ import annotations

from .ara import NamaaMrTydiReranking
from .eng import (
    AskUbuntuDupQuestions,
    FinFactReranking,
    FiQA2018Reranking,
    HC3Reranking,
    MindSmallReranking,
    NevIR,
    SciDocsReranking,
    StackOverflowDupQuestions,
    WebLINXCandidatesReranking,
)
from .fra import AlloprofReranking, SyntecReranking
from .jpn import VoyageMMarcoReranking
from .multilingual import ESCIReranking, MIRACLReranking, WikipediaRerankingMultilingual
from .rus import RuBQReranking
from .zho import (
    CMedQAv1,
    CMedQAv2,
    DISCFinLLMReranking,
    FinEvaReranking,
    MMarcoReranking,
    T2Reranking,
)

__all__ = [
    "CMedQAv1",
    "CMedQAv2",
    "MMarcoReranking",
    "T2Reranking",
    "NamaaMrTydiReranking",
    "AskUbuntuDupQuestions",
    "WebLINXCandidatesReranking",
    "StackOverflowDupQuestions",
    "NevIR",
    "MindSmallReranking",
    "SciDocsReranking",
    "VoyageMMarcoReranking",
    "MIRACLReranking",
    "ESCIReranking",
    "WikipediaRerankingMultilingual",
    "RuBQReranking",
    "SyntecReranking",
    "AlloprofReranking",
    "FiQA2018Reranking",
    "FinFactReranking",
    "HC3Reranking",
    "DISCFinLLMReranking",
    "FinEvaReranking",
]

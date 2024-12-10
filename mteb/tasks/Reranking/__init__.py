from __future__ import annotations

from .eng import (
    AskUbuntuDupQuestions,
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
from .zho import CMedQAv1, CMedQAv2, MMarcoReranking, T2Reranking

__all__ = [
    "CMedQAv1",
    "CMedQAv2",
    "MMarcoReranking",
    "T2Reranking",
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
]

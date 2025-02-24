from __future__ import annotations

from .ara import NamaaMrTydiReranking
from .eng import (
    AskUbuntuDupQuestions,
    BIRCOArguAnaReranking,
    BIRCOClinicalTrialReranking,
    BIRCODorisMaeReranking,
    BIRCORelicReranking,
    BIRCOWhatsThatBookReranking,
    BuiltBenchReranking,
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
    "NamaaMrTydiReranking",
    "BIRCODorisMaeReranking",
    "AskUbuntuDupQuestions",
    "WebLINXCandidatesReranking",
    "StackOverflowDupQuestions",
    "BIRCORelicReranking",
    "BIRCOArguAnaReranking",
    "NevIR",
    "MindSmallReranking",
    "BIRCOWhatsThatBookReranking",
    "BuiltBenchReranking",
    "BIRCOClinicalTrialReranking",
    "SciDocsReranking",
    "VoyageMMarcoReranking",
    "MIRACLReranking",
    "ESCIReranking",
    "WikipediaRerankingMultilingual",
    "RuBQReranking",
    "SyntecReranking",
    "AlloprofReranking",
]

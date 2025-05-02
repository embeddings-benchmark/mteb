from __future__ import annotations

from mteb.tasks.Retrieval.eng.BIRCOArguAnaReranking import BIRCOArguAnaReranking
from mteb.tasks.Retrieval.eng.BIRCOClinicalTrialReranking import (
    BIRCOClinicalTrialReranking,
)
from mteb.tasks.Retrieval.eng.BIRCODorisMaeReranking import BIRCODorisMaeReranking
from mteb.tasks.Retrieval.eng.BIRCORelicReranking import BIRCORelicReranking
from mteb.tasks.Retrieval.eng.BIRCOWhatsThatBookReranking import (
    BIRCOWhatsThatBookReranking,
)

from .AskUbuntuDupQuestions import AskUbuntuDupQuestions
from .BuiltBenchReranking import BuiltBenchReranking
from .MindSmallReranking import MindSmallReranking
from .NevIR import NevIR
from .SciDocsReranking import SciDocsReranking
from .StackOverflowDupQuestions import StackOverflowDupQuestions
from .WebLINXCandidatesReranking import WebLINXCandidatesReranking

__all__ = [
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
]

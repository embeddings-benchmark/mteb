from __future__ import annotations

from .IFIRAilaRetrieval import IFIRAila
from .IFIRCdsRetrieval import IFIRCds
from .IFIRFiQARetrieval import IFIRFiQA
from .IFIRFireRetrieval import IFIRFire
from .IFIRNFCorpusRetrieval import IFIRNFCorpus
from .IFIRPmRetrieval import IFIRPm
from .IFIRScifactRetrieval import IFIRScifact
from .InstructIR import InstructIR

__all__ = [
    "IFIRAila",
    "IFIRCds",
    "IFIRFiQA",
    "IFIRFire",
    "IFIRNFCorpus",
    "IFIRPm",
    "IFIRScifact",
    "InstructIR",
]

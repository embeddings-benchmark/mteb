from __future__ import annotations

from .deu import GermanSTSBenchmarkSTS
from .eng import (
    STS12STS,
    STS13STS,
    STS14STS,
    STS15STS,
    STS16STS,
    BiossesSTS,
    SickrSTS,
    STSBenchmarkSTS,
)
from .fao import FaroeseSTS
from .fas import Farsick, Query2Query, SynPerSTS
from .fin import FinParaSTS
from .fra import SickFrSTS
from .jpn import JSICK, JSTS
from .kor import KlueSTS, KorSTS
from .multilingual import (
    IndicCrosslingualSTS,
    SemRel24STS,
    STS17Crosslingual,
    STS22CrosslingualSTS,
    STS22CrosslingualSTSv2,
    STSBenchmarkMultilingualSTS,
)
from .pol import CdscrSTS, SickrPLSTS
from .por import Assin2STS, SickBrSTS
from .ron import RonSTS
from .rus import RUParaPhraserSTS, RuSTSBenchmarkSTS
from .spa import STSES
from .zho import AFQMC, ATEC, BQ, LCQMC, PAWSX, QBQTC, STSB

__all__ = [
    "AFQMC",
    "ATEC",
    "BQ",
    "LCQMC",
    "PAWSX",
    "QBQTC",
    "STSB",
    "Assin2STS",
    "SickBrSTS",
    "STSES",
    "CdscrSTS",
    "SickrPLSTS",
    "FinParaSTS",
    "GermanSTSBenchmarkSTS",
    "STS12STS",
    "STS13STS",
    "BiossesSTS",
    "STS15STS",
    "STSBenchmarkSTS",
    "SickrSTS",
    "STS16STS",
    "STS14STS",
    "Farsick",
    "Query2Query",
    "SynPerSTS",
    "FaroeseSTS",
    "JSICK",
    "JSTS",
    "RonSTS",
    "KorSTS",
    "KlueSTS",
    "IndicCrosslingualSTS",
    "SemRel24STS",
    "STS17Crosslingual",
    "STS22CrosslingualSTS",
    "STS22CrosslingualSTSv2",
    "STSBenchmarkMultilingualSTS",
    "RUParaPhraserSTS",
    "RuSTSBenchmarkSTS",
    "SickFrSTS",
]


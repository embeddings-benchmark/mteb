from __future__ import annotations

from .dan import BornholmBitextMining
from .eng import PubChemSMILESBitextMining
from .fas import SAMSumFa, SynPerChatbotRAGSumSRetrieval, SynPerChatbotSumSRetrieval
from .kat import TbilisiCityHallBitextMining
from .multilingual import (
    BibleNLPBitextMining,
    BUCCBitextMining,
    BUCCBitextMiningFast,
    DiaBLaBitextMining,
    FloresBitextMining,
    IN22ConvBitextMining,
    IN22GenBitextMining,
    IndicGenBenchFloresBitextMining,
    IWSLT2017BitextMining,
    LinceMTBitextMining,
    NollySentiBitextMining,
    NorwegianCourtsBitextMining,
    NTREXBitextMining,
    NusaTranslationBitextMining,
    NusaXBitextMining,
    PhincBitextMining,
    RomaTalesBitextMining,
    TatoebaBitextMining,
    WebFAQBitextMiningQAs,
    WebFAQBitextMiningQuestions,
)
from .srn import SRNCorpusBitextMining
from .vie import VieMedEVBitextMining

__all__ = [
    "BUCCBitextMining",
    "BUCCBitextMiningFast",
    "BibleNLPBitextMining",
    "BornholmBitextMining",
    "DiaBLaBitextMining",
    "FloresBitextMining",
    "IN22ConvBitextMining",
    "IN22GenBitextMining",
    "IWSLT2017BitextMining",
    "IndicGenBenchFloresBitextMining",
    "LinceMTBitextMining",
    "NTREXBitextMining",
    "NollySentiBitextMining",
    "NorwegianCourtsBitextMining",
    "NusaTranslationBitextMining",
    "NusaXBitextMining",
    "PhincBitextMining",
    "PubChemSMILESBitextMining",
    "RomaTalesBitextMining",
    "SAMSumFa",
    "SRNCorpusBitextMining",
    "SynPerChatbotRAGSumSRetrieval",
    "SynPerChatbotSumSRetrieval",
    "TatoebaBitextMining",
    "TbilisiCityHallBitextMining",
    "VieMedEVBitextMining",
    "WebFAQBitextMiningQAs",
    "WebFAQBitextMiningQuestions",
]

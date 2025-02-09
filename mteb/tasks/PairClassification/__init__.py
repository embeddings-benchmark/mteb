from __future__ import annotations

from .ara import ArEntail
from .ces import CTKFactsNLI
from .deu import FalseFriendsDeEnPC
from .eng import (
    HeadlineACPairClassification,
    HeadlinePDDPairClassification,
    HeadlinePDUPairClassification,
    LegalBenchPC,
    PubChemAISentenceParaphrasePC,
    PubChemSMILESPC,
    PubChemSynonymPC,
    PubChemWikiParagraphsPC,
    SprintDuplicateQuestionsPC,
    TwitterSemEval2015PC,
    TwitterURLCorpusPC,
)
from .fas import (
    CExaPPC,
    FarsiParaphraseDetection,
    FarsTail,
    ParsinluEntail,
    ParsinluQueryParaphPC,
    SynPerChatbotRAGFAQPC,
    SynPerQAPC,
    SynPerTextKeywordsPC,
)
from .hye import ArmenianParaphrasePC
from .ind import IndoNLI
from .kor import KlueNLI
from .multilingual import (
    RTE3,
    XNLI,
    XNLIV2,
    IndicXnliPairClassification,
    OpusparcusPC,
    PawsXPairClassification,
    PubChemWikiPairClassification,
    XStance,
)
from .pol import CdscePC, PpcPC, PscPC, SickePLPC
from .por import Assin2RTE, SickBrPC
from .rus import TERRa
from .zho import AFQMCPairClassification, Cmnli, Ocnli

__all__ = [
    "Cmnli",
    "Ocnli",
    "Assin2RTE",
    "SickBrPC",
    "CdscePC",
    "PpcPC",
    "PscPC",
    "SickePLPC",
    "IndoNLI",
    "FalseFriendsDeEnPC",
    "ArEntail",
    "ArmenianParaphrasePC",
    "CTKFactsNLI",
    "PubChemSMILESPC",
    "PubChemSynonymPC",
    "LegalBenchPC",
    "TwitterSemEval2015PC",
    "PubChemWikiParagraphsPC",
    "TwitterURLCorpusPC",
    "SprintDuplicateQuestionsPC",
    "PubChemAISentenceParaphrasePC",
    "FarsTail",
    "CExaPPC",
    "FarsiParaphraseDetection",
    "ParsinluEntail",
    "ParsinluQueryParaphPC",
    "SynPerChatbotRAGFAQPC",
    "SynPerQAPC",
    "SynPerTextKeywordsPC",
    "KlueNLI",
    "IndicXnliPairClassification",
    "OpusparcusPC",
    "PawsXPairClassification",
    "RTE3",
    "PubChemWikiPairClassification",
    "XStance",
    "XNLI",
    "XNLIV2",
    "TERRa",
    "HeadlineACPairClassification",
    "HeadlinePDDPairClassification",
    "HeadlinePDUPairClassification",
    "AFQMCPairClassification",
]

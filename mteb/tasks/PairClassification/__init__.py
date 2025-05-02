from __future__ import annotations

from .ara import ArEntail
from .ces import CTKFactsNLI
from .deu import FalseFriendsDeEnPC
from .eng import (
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
from .zho import Cmnli, Ocnli

__all__ = [
    "ArEntail",
    "ArmenianParaphrasePC",
    "Assin2RTE",
    "CExaPPC",
    "CTKFactsNLI",
    "CdscePC",
    "Cmnli",
    "FalseFriendsDeEnPC",
    "FarsTail",
    "FarsiParaphraseDetection",
    "IndicXnliPairClassification",
    "IndoNLI",
    "KlueNLI",
    "LegalBenchPC",
    "Ocnli",
    "OpusparcusPC",
    "ParsinluEntail",
    "ParsinluQueryParaphPC",
    "PawsXPairClassification",
    "PpcPC",
    "PscPC",
    "PubChemAISentenceParaphrasePC",
    "PubChemSMILESPC",
    "PubChemSynonymPC",
    "PubChemWikiPairClassification",
    "PubChemWikiParagraphsPC",
    "RTE3",
    "SickBrPC",
    "SickePLPC",
    "SprintDuplicateQuestionsPC",
    "SynPerChatbotRAGFAQPC",
    "SynPerQAPC",
    "SynPerTextKeywordsPC",
    "TERRa",
    "TwitterSemEval2015PC",
    "TwitterURLCorpusPC",
    "XNLI",
    "XNLIV2",
    "XStance",
]

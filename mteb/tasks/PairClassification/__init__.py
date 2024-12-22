from __future__ import annotations

from .ara import ArEntail
from .ces import CTKFactsNLI
from .deu import FalseFriendsDeEnPC
from .eng import (
    LegalBenchPC,
    SprintDuplicateQuestionsPC,
    TwitterSemEval2015PC,
    TwitterURLCorpusPC,
)
from .fas import FarsTail
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
    XStance,
)
from .pol import CdscePC, PpcPC, PscPC, SickePLPC
from .por import Assin2RTE, SickBrPC
from .rus import TERRa
from .zho import Cmnli, Ocnli

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
    "LegalBenchPC",
    "TwitterSemEval2015PC",
    "TwitterURLCorpusPC",
    "SprintDuplicateQuestionsPC",
    "FarsTail",
    "KlueNLI",
    "IndicXnliPairClassification",
    "OpusparcusPC",
    "PawsXPairClassification",
    "RTE3",
    "XStance",
    "XNLI",
    "XNLIV2",
    "TERRa",
]

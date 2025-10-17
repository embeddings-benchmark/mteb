from .indic_xnli_pair_classification import IndicXnliPairClassification
from .opusparcus_pc import OpusparcusPC
from .paws_x_pair_classification import PawsXPairClassification
from .pub_chem_wiki_pair_classification import PubChemWikiPairClassification
from .rte3 import RTE3
from .x_stance import XStance
from .xnli import XNLI, XNLIV2

__all__ = [
    "RTE3",
    "XNLI",
    "XNLIV2",
    "IndicXnliPairClassification",
    "OpusparcusPC",
    "PawsXPairClassification",
    "PubChemWikiPairClassification",
    "XStance",
]

from .ArguAnaPLRetrieval import ArguAnaPL
from .CqadupstackPLRetrieval import (
    CQADupstackAndroidRetrievalPL,
    CQADupstackEnglishRetrievalPL,
    CQADupstackGamingRetrievalPL,
    CQADupstackGisRetrievalPL,
    CQADupstackMathematicaRetrievalPL,
    CQADupstackPhysicsRetrievalPL,
    CQADupstackProgrammersRetrievalPL,
    CQADupstackStatsRetrievalPL,
    CQADupstackTexRetrievalPL,
    CQADupstackUnixRetrievalPL,
    CQADupstackWebmastersRetrievalPL,
    CQADupstackWordpressRetrievalPL,
)
from .DBPediaPLRetrieval import DBPediaPL, DBPediaPLHardNegatives
from .FiQAPLRetrieval import FiQAPLRetrieval
from .HotpotQAPLRetrieval import HotpotQAPL, HotpotQAPLHardNegatives
from .MSMARCOPLRetrieval import MSMARCOPL, MSMARCOPLHardNegatives
from .NFCorpusPLRetrieval import NFCorpusPL
from .NQPLRetrieval import NQPL, NQPLHardNegatives
from .PUGGRetrieval import PUGGRetrieval
from .QuoraPLRetrieval import QuoraPLRetrieval, QuoraPLRetrievalHardNegatives
from .SCIDOCSPLRetrieval import SCIDOCSPL
from .SciFactPLRetrieval import SciFactPL
from .Touche2020PLRetrieval import Touche2020PL
from .TRECCOVIDPLRetrieval import TRECCOVIDPL

__all__ = [
    "MSMARCOPL",
    "NQPL",
    "SCIDOCSPL",
    "TRECCOVIDPL",
    "ArguAnaPL",
    "CQADupstackAndroidRetrievalPL",
    "CQADupstackEnglishRetrievalPL",
    "CQADupstackGamingRetrievalPL",
    "CQADupstackGisRetrievalPL",
    "CQADupstackMathematicaRetrievalPL",
    "CQADupstackPhysicsRetrievalPL",
    "CQADupstackProgrammersRetrievalPL",
    "CQADupstackStatsRetrievalPL",
    "CQADupstackTexRetrievalPL",
    "CQADupstackUnixRetrievalPL",
    "CQADupstackWebmastersRetrievalPL",
    "CQADupstackWordpressRetrievalPL",
    "DBPediaPL",
    "DBPediaPLHardNegatives",
    "FiQAPLRetrieval",
    "HotpotQAPL",
    "HotpotQAPLHardNegatives",
    "MSMARCOPLHardNegatives",
    "NFCorpusPL",
    "NQPLHardNegatives",
    "PUGGRetrieval",
    "QuoraPLRetrieval",
    "QuoraPLRetrievalHardNegatives",
    "SciFactPL",
    "Touche2020PL",
]

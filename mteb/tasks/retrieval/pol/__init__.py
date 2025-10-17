from .argu_ana_pl_retrieval import ArguAnaPL
from .cqadupstack_pl_retrieval import (
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
from .db_pedia_pl_retrieval import DBPediaPL, DBPediaPLHardNegatives
from .fi_qapl_retrieval import FiQAPLRetrieval
from .hotpot_qapl_retrieval import HotpotQAPL, HotpotQAPLHardNegatives
from .msmarcopl_retrieval import MSMARCOPL, MSMARCOPLHardNegatives
from .nf_corpus_pl_retrieval import NFCorpusPL
from .nqpl_retrieval import NQPL, NQPLHardNegatives
from .pugg_retrieval import PUGGRetrieval
from .quora_pl_retrieval import QuoraPLRetrieval, QuoraPLRetrievalHardNegatives
from .sci_fact_pl_retrieval import SciFactPL
from .scidocspl_retrieval import SCIDOCSPL
from .touche2020_pl_retrieval import Touche2020PL
from .treccovidpl_retrieval import TRECCOVIDPL

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

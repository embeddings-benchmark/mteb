from __future__ import annotations

from .ArguAnaPLRetrieval import ArguAnaPL
from .DBPediaPLRetrieval import DBPediaPL, DBPediaPLHardNegatives
from .FiQAPLRetrieval import FiQAPLRetrieval
from .HotpotQAPLRetrieval import HotpotQAPL, HotpotQAPLHardNegatives
from .MSMARCOPLRetrieval import MSMARCOPL, MSMARCOPLHardNegatives
from .NFCorpusPLRetrieval import NFCorpusPL
from .NQPLRetrieval import NQPL, NQPLHardNegatives
from .QuoraPLRetrieval import QuoraPLRetrieval, QuoraPLRetrievalHardNegatives
from .SCIDOCSPLRetrieval import SCIDOCSPL
from .SciFactPLRetrieval import SciFactPL
from .TRECCOVIDPLRetrieval import TRECCOVIDPL

__all__ = [
    "MSMARCOPL",
    "MSMARCOPLHardNegatives",
    "SCIDOCSPL",
    "SciFactPL",
    "ArguAnaPL",
    "FiQAPLRetrieval",
    "NFCorpusPL",
    "QuoraPLRetrieval",
    "QuoraPLRetrievalHardNegatives",
    "TRECCOVIDPL",
    "NQPL",
    "NQPLHardNegatives",
    "DBPediaPL",
    "DBPediaPLHardNegatives",
    "HotpotQAPL",
    "HotpotQAPLHardNegatives",
]

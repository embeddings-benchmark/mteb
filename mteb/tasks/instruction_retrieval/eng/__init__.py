from .ifir_aila_retrieval import IFIRAila
from .ifir_cds_retrieval import IFIRCds
from .ifir_fi_qa_retrieval import IFIRFiQA
from .ifir_fire_retrieval import IFIRFire
from .ifir_pm_retrieval import IFIRPm
from .ifir_scifact_retrieval import IFIRScifact
from .ifirnf_corpus_retrieval import IFIRNFCorpus
from .instruct_ir import InstructIR

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

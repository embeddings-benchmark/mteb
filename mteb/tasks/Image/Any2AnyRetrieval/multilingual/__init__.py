from __future__ import annotations

from .MIRACLVisionRetrieval import MIRACLVisionRetrieval
from .VdrMultilingualRetrieval import VDRMultilingualRetrieval
from .Vidore2BenchRetrieval import (
    Vidore2BioMedicalLecturesRetrieval,
    Vidore2EconomicsReportsRetrieval,
    Vidore2ESGReportsHLRetrieval,
    Vidore2ESGReportsRetrieval,
)
from .WITT2IRetrieval import WITT2IRetrieval
from .XFlickr30kCoT2IRetrieval import XFlickr30kCoT2IRetrieval
from .XM3600T2IRetrieval import XM3600T2IRetrieval

__all__ = [
    "Vidore2BioMedicalLecturesRetrieval",
    "Vidore2ESGReportsHLRetrieval",
    "Vidore2ESGReportsRetrieval",
    "Vidore2EconomicsReportsRetrieval",
    "WITT2IRetrieval",
    "XFlickr30kCoT2IRetrieval",
    "XM3600T2IRetrieval",
    "VDRMultilingualRetrieval",
    "MIRACLVisionRetrieval",
]

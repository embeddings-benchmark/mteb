from __future__ import annotations

from .IndicCrosslingualSTS import IndicCrosslingualSTS
from .SemRel24STS import SemRel24STS
from .STS17CrosslingualSTS import STS17Crosslingual
from .STS22CrosslingualSTS import STS22CrosslingualSTS, STS22CrosslingualSTSv2
from .STSBenchmarkMultilingualSTS import STSBenchmarkMultilingualSTS

__all__ = [
    "IndicCrosslingualSTS",
    "SemRel24STS",
    "STS17Crosslingual",
    "STS22CrosslingualSTS",
    "STS22CrosslingualSTSv2",
    "STSBenchmarkMultilingualSTS",
]

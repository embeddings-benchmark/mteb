from .humests22 import HUMESTS22
from .indic_crosslingual_sts import IndicCrosslingualSTS
from .sem_rel24_sts import SemRel24STS
from .sts17_crosslingual_sts import STS17Crosslingual
from .sts17_multilingual_visual_sts import STS17MultilingualVisualSTS
from .sts22_crosslingual_sts import STS22CrosslingualSTS, STS22CrosslingualSTSv2
from .sts_benchmark_multilingual_sts import STSBenchmarkMultilingualSTS
from .sts_benchmark_multilingual_visual_sts import STSBenchmarkMultilingualVisualSTS

__all__ = [
    "HUMESTS22",
    "IndicCrosslingualSTS",
    "STS17Crosslingual",
    "STS17MultilingualVisualSTS",
    "STS22CrosslingualSTS",
    "STS22CrosslingualSTSv2",
    "STSBenchmarkMultilingualSTS",
    "STSBenchmarkMultilingualVisualSTS",
    "SemRel24STS",
]

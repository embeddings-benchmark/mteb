from __future__ import annotations

from .en import (
    STS12VisualSTS,
    STS13VisualSTS,
    STS14VisualSTS,
    STS15VisualSTS,
    STS16VisualSTS,
)
from .multilingual import STS17MultilingualVisualSTS, STSBenchmarkMultilingualVisualSTS

__all__ = [
    "STS17MultilingualVisualSTS",
    "STSBenchmarkMultilingualVisualSTS",
    "STS13VisualSTS",
    "STS15VisualSTS",
    "STS12VisualSTS",
    "STS16VisualSTS",
    "STS14VisualSTS",
]

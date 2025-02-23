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
    "STS12VisualSTS",
    "STS13VisualSTS",
    "STS14VisualSTS",
    "STS15VisualSTS",
    "STS16VisualSTS",
    "STS17MultilingualVisualSTS",
    "STSBenchmarkMultilingualVisualSTS",
]

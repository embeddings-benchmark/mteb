from __future__ import annotations

from .BLINKIT2IMultiChoice import BLINKIT2IMultiChoice
from .BLINKIT2TMultiChoice import BLINKIT2TMultiChoice
from .ImageCoDeT2IMultiChoice import ImageCoDeT2IMultiChoice
from .ROxfordI2IMultiChoice import (
    ROxfordEasyI2IMultiChoice,
    ROxfordHardI2IMultiChoice,
    ROxfordMediumI2IMultiChoice,
)
from .RParisI2IMultiChoice import (
    RParisEasyI2IMultiChoice,
    RParisHardI2IMultiChoice,
    RParisMediumI2IMultiChoice,
)

__all__ = [
    "ImageCoDeT2IMultiChoice",
    "BLINKIT2IMultiChoice",
    "BLINKIT2TMultiChoice",
    "ROxfordEasyI2IMultiChoice",
    "ROxfordHardI2IMultiChoice",
    "ROxfordMediumI2IMultiChoice",
    "RParisEasyI2IMultiChoice",
    "RParisHardI2IMultiChoice",
    "RParisMediumI2IMultiChoice",
]

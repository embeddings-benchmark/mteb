from __future__ import annotations

from .eng import (
    Core17InstructionRetrieval,
    News21InstructionRetrieval,
    Robust04InstructionRetrieval,
)
from .multilingual import mFollowIR, mFollowIRCrossLingual

__all__ = [
    "Core17InstructionRetrieval",
    "News21InstructionRetrieval",
    "Robust04InstructionRetrieval",
    "mFollowIR",
    "mFollowIRCrossLingual",
]

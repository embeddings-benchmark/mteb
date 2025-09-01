from __future__ import annotations

from .DalajClassification import DalajClassification, DalajClassificationV2
from .SwedishSentimentClassification import (
    SwedishSentimentClassification,
    SwedishSentimentClassificationV2,
)
from .SweRecClassification import SweRecClassification, SweRecClassificationV2

__all__ = [
    "DalajClassification",
    "DalajClassificationV2",
    "SweRecClassification",
    "SweRecClassificationV2",
    "SwedishSentimentClassification",
    "SwedishSentimentClassificationV2",
]

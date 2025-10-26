from .audio_set import (
    AudioSetMiniMultilingualClassification,
    AudioSetMultilingualClassification,
)
from .bird_set import BirdSetMultilabelClassification
from .fsd50_hf import FSD50HFMultilingualClassification
from .fsd2019_kaggle import FSD2019KaggleMultilingualClassification

__all__ = [
    "AudioSetMiniMultilingualClassification",
    "AudioSetMultilingualClassification",
    "BirdSetMultilabelClassification",
    "FSD50HFMultilingualClassification",
    "FSD2019KaggleMultilingualClassification",
]

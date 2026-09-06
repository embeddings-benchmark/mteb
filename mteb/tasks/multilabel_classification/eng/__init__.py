from .audio_set import (
    AudioSetMiniMultilingualClassification,
    AudioSetMultilingualClassification,
)
from .fsd50_hf import FSD50HFMultilingualClassification
from .fsd2019_kaggle import FSD2019KaggleMultilingualClassification
from .pascal_voc2007 import VOC2007Classification
from .scirepeval_fos_classification import SciRepEvalFoSClassification

__all__ = [
    "AudioSetMiniMultilingualClassification",
    "AudioSetMultilingualClassification",
    "FSD50HFMultilingualClassification",
    "FSD2019KaggleMultilingualClassification",
    "SciRepEvalFoSClassification",
    "VOC2007Classification",
]

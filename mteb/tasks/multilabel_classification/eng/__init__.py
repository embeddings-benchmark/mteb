from .audio_set import (
    AudioSetMiniMultilingualClassification,
    AudioSetMultilingualClassification,
)
from .fsd50_hf import FSD50HFMultilingualClassification
from .fsd2019_kaggle import (
    FSD2019KaggleMultilingualClassification,
    FSD2019KaggleMultilingualClassificationV2,
)
from .pascal_voc2007 import (
    VOC2007Classification,
    VOC2007ClassificationV2,
)
from .scirepeval_fos_classification import SciRepEvalFoSClassification

__all__ = [
    "AudioSetMiniMultilingualClassification",
    "AudioSetMultilingualClassification",
    "FSD50HFMultilingualClassification",
    "FSD2019KaggleMultilingualClassification",
    "FSD2019KaggleMultilingualClassificationV2",
    "SciRepEvalFoSClassification",
    "VOC2007Classification",
    "VOC2007ClassificationV2",
]

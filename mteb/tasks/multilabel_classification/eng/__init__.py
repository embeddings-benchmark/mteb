from .audio_set import (
    AudioSetMiniMultilingualClassification,
    AudioSetMiniMultilingualClassificationV2,
    AudioSetMultilingualClassification,
)
from .fsd50_hf import (
    FSD50HFMultilingualClassification,
    FSD50HFMultilingualClassificationV2,
)
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
    "AudioSetMiniMultilingualClassificationV2",
    "AudioSetMultilingualClassification",
    "FSD50HFMultilingualClassification",
    "FSD50HFMultilingualClassificationV2",
    "FSD2019KaggleMultilingualClassification",
    "FSD2019KaggleMultilingualClassificationV2",
    "SciRepEvalFoSClassification",
    "VOC2007Classification",
    "VOC2007ClassificationV2",
]

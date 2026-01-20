from .common_language_age_detection import CommonLanguageAgeDetection
from .common_language_gender_detection import CommonLanguageGenderDetection
from .common_language_language_classification import (
    CommonLanguageLanguageClassification,
)
from .cremad import CREMAD
from .cstr_vctk_accent_id import CSTRVCTKAccentID
from .cstr_vctk_gender_classification import CSTRVCTKGenderClassification
from .expresso import (
    ExpressoConvEmotionClassification,
    ExpressoReadEmotionClassification,
)
from .fsdd import FSDD
from .globe_v2_age_classification import GlobeV2AgeClassification
from .globe_v2_gender_classification import GlobeV2GenderClassification
from .iemocap_emotion import IEMOCAPEmotionClassification
from .iemocap_gender import IEMOCAPGenderClassification
from .libri_count import LibriCount
from .speech_commands import SpeechCommandsClassification
from .spoke_n import SpokeNEnglishClassification
from .spoken_q_afor_ic import SpokenQAForIC
from .vocal_sound import VocalSoundClassification
from .vox_celeb_sa import VoxCelebSA
from .vox_lingua107_top10 import VoxLingua107Top10
from .vox_populi_accent_id import VoxPopuliAccentID

__all__ = [
    "CREMAD",
    "FSDD",
    "CSTRVCTKAccentID",
    "CSTRVCTKGenderClassification",
    "CommonLanguageAgeDetection",
    "CommonLanguageGenderDetection",
    "CommonLanguageLanguageClassification",
    "ExpressoConvEmotionClassification",
    "ExpressoReadEmotionClassification",
    "GlobeV2AgeClassification",
    "GlobeV2GenderClassification",
    "IEMOCAPEmotionClassification",
    "IEMOCAPGenderClassification",
    "LibriCount",
    "SpeechCommandsClassification",
    "SpokeNEnglishClassification",
    "SpokenQAForIC",
    "VocalSoundClassification",
    "VoxCelebSA",
    "VoxLingua107Top10",
    "VoxPopuliAccentID",
]

from .ambient_acoustic_context import AmbientAcousticContextClassification
from .beijing_opera import BeijingOpera
from .bird_clef import BirdCLEFClassification
from .common_language_age_detection import CommonLanguageAgeDetection
from .common_language_gender_detection import CommonLanguageGenderDetection
from .common_language_language_classification import (
    CommonLanguageLanguageClassification,
)
from .cremad import CREMAD
from .esc50 import ESC50Classification
from .fsdd import FSDD
from .gtzan_genre import GTZANGenre
from .gunshot_triangulation import GunshotTriangulation
from .iemocap_emotion import IEMOCAPEmotionClassification
from .iemocap_gender import IEMOCAPGenderClassification
from .libri_count import LibriCount
from .mridingham_stroke import MridinghamStroke
from .mridingham_tonic import MridinghamTonic
from .n_synth import NSynth
from .speech_commands import SpeechCommandsClassification
from .spoke_n import SpokeNEnglishClassification
from .spoken_q_afor_ic import SpokenQAforIC
from .tut_acoustic_scenes import TUTAcousticScenesClassification
from .urban_sound8k import UrbanSound8kZeroshotClassification
from .vocal_sound import VocalSoundClassification
from .vox_celeb_sa import VoxCelebSA
from .vox_lingua107_top10 import VoxLingua107Top10
from .vox_populi_accent_id import VoxPopuliAccentID

__all__ = [
    "CREMAD",
    "FSDD",
    "AmbientAcousticContextClassification",
    "BeijingOpera",
    "BirdCLEFClassification",
    "CommonLanguageAgeDetection",
    "CommonLanguageGenderDetection",
    "CommonLanguageLanguageClassification",
    "ESC50Classification",
    "GTZANGenre",
    "GunshotTriangulation",
    "IEMOCAPEmotionClassification",
    "IEMOCAPGenderClassification",
    "LibriCount",
    "MridinghamStroke",
    "MridinghamTonic",
    "NSynth",
    "SpeechCommandsClassification",
    "SpokeNEnglishClassification",
    "SpokenQAforIC",
    "TUTAcousticScenesClassification",
    "UrbanSound8kZeroshotClassification",
    "VocalSoundClassification",
    "VoxCelebSA",
    "VoxLingua107Top10",
    "VoxPopuliAccentID",
]

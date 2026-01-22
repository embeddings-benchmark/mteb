from .audio_caps import AudioCapsA2TRetrieval, AudioCapsT2ARetrieval
from .common_voice import (
    CommonVoiceMini17A2TRetrieval,
    CommonVoiceMini17T2ARetrieval,
    CommonVoiceMini21A2TRetrieval,
    CommonVoiceMini21T2ARetrieval,
)
from .fleurs import FleursA2TRetrieval, FleursT2ARetrieval
from .google_svq import GoogleSVQA2TRetrieval, GoogleSVQT2ARetrieval
from .jam_alt import (
    JamAltArtistA2ARetrieval,
    JamAltLyricA2TRetrieval,
    JamAltLyricT2ARetrieval,
)

__all__ = [
    "AudioCapsA2TRetrieval",
    "AudioCapsT2ARetrieval",
    "CommonVoiceMini17A2TRetrieval",
    "CommonVoiceMini17T2ARetrieval",
    "CommonVoiceMini21A2TRetrieval",
    "CommonVoiceMini21T2ARetrieval",
    "FleursA2TRetrieval",
    "FleursT2ARetrieval",
    "GoogleSVQA2TRetrieval",
    "GoogleSVQT2ARetrieval",
    "JamAltArtistA2ARetrieval",
    "JamAltLyricA2TRetrieval",
    "JamAltLyricT2ARetrieval",
]

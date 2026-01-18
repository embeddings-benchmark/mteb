from .audio_caps import AudioCapsA2TRetrieval, AudioCapsT2ARetrieval
from .common_voice import (
    MiniCommonVoice17A2TRetrieval,
    MiniCommonVoice17T2ARetrieval,
    MiniCommonVoice21A2TRetrieval,
    MiniCommonVoice21T2ARetrieval,
)
from .fleurs import FleursA2TRetrieval, FleursT2ARetrieval
from .google_svq import GoogleSVQA2TRetrieval, GoogleSVQT2ARetrieval
from .jam_alt import JamAltArtist, JamAltLyricsA2T, JamAltLyricsT2A

__all__ = [
    "AudioCapsA2TRetrieval",
    "AudioCapsT2ARetrieval",
    "FleursA2TRetrieval",
    "FleursT2ARetrieval",
    "GoogleSVQA2TRetrieval",
    "GoogleSVQT2ARetrieval",
    "JamAltArtist",
    "JamAltLyricsA2T",
    "JamAltLyricsT2A",
    "MiniCommonVoice17A2TRetrieval",
    "MiniCommonVoice17T2ARetrieval",
    "MiniCommonVoice21A2TRetrieval",
    "MiniCommonVoice21T2ARetrieval",
]

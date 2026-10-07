"""Every audio wrapper must reject a non-positive length cap before loading weights.

VGGish and YAMNet define their wrappers inside a loader function, so they are
not importable here; they run the same check first in __init__.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

# (module, class, cap keyword)
CASES = [
    ("audio_flamingo", "AudioFlamingoWrapper", "max_audio_length_seconds"),
    ("audio_qwen2_models", "Qwen2AudioWrapper", "max_audio_length_seconds"),
    ("bidirlm_omni_models", "BidirLMOmniEncoder", "max_samples"),
    ("cnn14_model", "CNN14Wrapper", "max_audio_length_seconds"),
    ("colqwen_models", "ColQwen2_5OmniWrapper", "max_audio_length"),
    ("data2vec_models", "Data2VecAudioWrapper", "max_audio_length_seconds"),
    ("e5_omni_models", "E5OmniWrapper", "max_samples"),
    ("encodec_model", "EncodecWrapper", "max_audio_length_seconds"),
    ("hubert_models", "HubertWrapper", "max_audio_length_seconds"),
    ("language_bind_models", "LanguageBindVideoWrapper", "max_samples"),
    ("lco_embedding_models", "LCOEmbedding", "max_audio_length"),
    ("mctct_model", "MCTCTWrapper", "max_audio_length_seconds"),
    ("mms_models", "MMSWrapper", "max_audio_length_seconds"),
    ("msclap_models", "MSClapWrapper", "max_audio_length_seconds"),
    ("muq_mulan_model", "MuQMuLanWrapper", "max_audio_length_seconds"),
    ("omni_embed_nemotron_models", "OmniEmbedNemotronWrapper", "max_audio_length"),
    ("omnivinci_models", "OmniVinciWrapper", "max_audio_length_seconds"),
    ("pe_av_models", "PEAudioVisualWrapper", "max_samples"),
    ("qwen3_voice_models", "Qwen3VoiceEmbeddingWrapper", "max_audio_length_seconds"),
    ("qwen_omni_lm", "QwenOmniWrapper", "max_audio_length_seconds"),
    ("seamlessm4t_models", "SeamlessM4TWrapper", "max_audio_length_seconds"),
    ("sewd_models", "SewDWrapper", "max_audio_length_seconds"),
    ("speecht5_models", "SpeechT5Audio", "max_audio_length_seconds"),
    ("tevatron_omni_embed_models", "TevatronOmniEmbedWrapper", "max_samples"),
    ("unispeech_models", "UniSpeechWrapper", "max_audio_length_seconds"),
    ("voiceclap_models", "VoiceCLAPSmallWrapper", "max_audio_length_seconds"),
    ("wav2clip_model", "Wav2ClipZeroShotWrapper", "max_audio_length_seconds"),
    ("wav2vec2_models", "Wav2Vec2AudioWrapper", "max_audio_length_seconds"),
    ("wavlm_models", "WavlmWrapper", "max_audio_length_seconds"),
    ("whisper_models", "WhisperAudioWrapper", "max_audio_length_seconds"),
]


@pytest.mark.parametrize(
    ("module", "cls_name", "cap_kw"), CASES, ids=[c[1] for c in CASES]
)
@pytest.mark.parametrize("cap", [0, -1])
def test_rejects_non_positive_cap(
    module: str, cls_name: str, cap_kw: str, cap: int
) -> None:
    mod = importlib.import_module(f"mteb.models.model_implementations.{module}")
    cls = getattr(mod, cls_name)
    params = inspect.signature(cls.__init__).parameters
    assert cap_kw in params, f"{cls_name} has no {cap_kw}"
    first = next(p for p in params if p != "self")
    kwargs = {first: "not-a-real-checkpoint", cap_kw: cap}
    if "revision" in params:
        kwargs["revision"] = "unused"
    with pytest.raises(ValueError, match="must be positive"):
        cls(**kwargs)

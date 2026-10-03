"""A clip's embedding must not depend on what else is in its batch.

Group-norm speech encoders (and data2vec) normalise over the padded batch, so
batching them changes the embeddings of shorter clips. These tests use tiny
random checkpoints with the same feature-encoder norms as the real ones.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from mteb._create_dataloaders import _custom_collate_fn

pytest.importorskip("torchaudio", reason="Audio dependencies are not installed")

from mteb.models.model_implementations.data2vec_models import Data2VecAudioWrapper
from mteb.models.model_implementations.hubert_models import HubertWrapper
from mteb.models.model_implementations.sewd_models import SewDWrapper
from mteb.models.model_implementations.unispeech_models import UniSpeechWrapper
from mteb.models.model_implementations.wav2vec2_models import Wav2Vec2AudioWrapper
from mteb.models.model_implementations.wavlm_models import WavlmWrapper

SAMPLING_RATE = 16_000

TINY_CHECKPOINTS = [
    # group norm
    (
        Wav2Vec2AudioWrapper,
        "hf-internal-testing/tiny-random-Wav2Vec2Model",
        "7a998ee3ee0619a52828a79c3eed6872fd053f37",
    ),
    (
        HubertWrapper,
        "hf-internal-testing/tiny-random-HubertModel",
        "3224562c86c4669db65ae7defdc5fb555b113e95",
    ),
    (
        WavlmWrapper,
        "hf-internal-testing/tiny-random-WavLMModel",
        "e932275e37cb643be271f655bd1d649f4f4b4bd5",
    ),
    (
        UniSpeechWrapper,
        "hf-internal-testing/tiny-random-UniSpeechSatForCTC",
        "a8617538d3a2ae990f022bb0c36b8428a4870822",
    ),
    (
        SewDWrapper,
        "hf-internal-testing/tiny-random-SEWDForCTC",
        "5c7495c77ae9e0f12c0de05d3a5fb95bdcd91768",
    ),
    # layer norm, but padding leaks through the positional convolution
    (
        Data2VecAudioWrapper,
        "hf-internal-testing/tiny-random-Data2VecAudioModel",
        "73f503fdff73b7616154f64dbe38a685cc48e8eb",
    ),
    # layer norm: batched with an attention mask, which is exact
    (
        Wav2Vec2AudioWrapper,
        "hf-internal-testing/tiny-random-wav2vec2",
        "9123c4c809823cc53466e9868a1cf1c476be2e54",
    ),
]


class _AudioDataset(Dataset):
    features = {"audio": None}

    def __init__(self, clips: list[np.ndarray]) -> None:
        self.clips = clips

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> dict:
        return {"audio": {"array": self.clips[i], "sampling_rate": SAMPLING_RATE}}


def _embed(model, clips: list[np.ndarray]) -> np.ndarray:
    loader = DataLoader(
        _AudioDataset(clips), batch_size=len(clips), collate_fn=_custom_collate_fn
    )
    return np.asarray(model.get_audio_embeddings(loader, show_progress_bar=False))


@pytest.mark.parametrize(
    ("wrapper", "name", "revision"),
    TINY_CHECKPOINTS,
    ids=[name.split("/")[-1] for _, name, _ in TINY_CHECKPOINTS],
)
def test_embedding_independent_of_batch_mates(wrapper, name, revision) -> None:
    torch.manual_seed(0)
    model = wrapper(model_name=name, revision=revision, device="cpu")
    rng = np.random.default_rng(0)
    short = rng.standard_normal(SAMPLING_RATE).astype(np.float32) * 0.1
    long = rng.standard_normal(SAMPLING_RATE * 4).astype(np.float32) * 0.1

    alone = _embed(model, [short])[0]
    batched = _embed(model, [short, long])[0]

    np.testing.assert_allclose(alone, batched, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    ("name", "revision", "per_clip"),
    [
        (
            "hf-internal-testing/tiny-random-wav2vec2",
            "9123c4c809823cc53466e9868a1cf1c476be2e54",
            False,
        ),
        (
            "hf-internal-testing/tiny-random-Wav2Vec2Model",
            "7a998ee3ee0619a52828a79c3eed6872fd053f37",
            True,
        ),
    ],
)
def test_only_group_norm_checkpoints_drop_batching(name, revision, per_clip) -> None:
    model = Wav2Vec2AudioWrapper(model_name=name, revision=revision, device="cpu")
    assert model.per_clip is per_clip

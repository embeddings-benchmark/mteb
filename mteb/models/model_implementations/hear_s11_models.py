from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType
    from mteb.types._encoder_io import AudioInput


class HeARS11AudioWrapper(AbsEncoder):
    sampling_rate = 16_000
    clip_samples = 32_000

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    @classmethod
    def _prepare_audio(cls, audio: np.ndarray) -> torch.Tensor:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        audio = audio.reshape(-1)

        if audio.shape[0] > cls.clip_samples:
            start = (audio.shape[0] - cls.clip_samples) // 2
            audio = audio[start : start + cls.clip_samples]
        elif audio.shape[0] < cls.clip_samples:
            audio = np.pad(audio, (0, cls.clip_samples - audio.shape[0]))

        return torch.from_numpy(audio)

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        inputs.collate_fn = AudioCollator(target_sampling_rate=self.sampling_rate)
        embeddings = []

        for batch in tqdm(inputs, disable=not show_progress_bar):
            audio = torch.stack(
                [self._prepare_audio(item["array"]) for item in batch["audio"]]
            ).to(self.device)

            with torch.no_grad():
                output = self.model(input_values=audio, return_dict=True)

            embeddings.append(output.pooler_output.cpu().detach())

        return torch.cat(embeddings, dim=0).numpy()

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        return self.get_audio_embeddings(inputs, **kwargs)


hear_s11_audio = ModelMeta(
    loader=HeARS11AudioWrapper,
    name="matthewagi/HeAR-s1.1",
    languages=None,
    open_weights=True,
    revision="a5776bebff935a81c79720467ae1e10a4effe10e",
    release_date="2026-03-26",
    max_tokens=None,
    n_parameters=22_140_288,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    embed_dim=384,
    license="https://developers.google.com/health-ai-developer-foundations/terms",
    reference="https://huggingface.co/matthewagi/HeAR-s1.1",
    similarity_fn_name="cosine",
    framework=["PyTorch", "Transformers", "safetensors"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=set(),
    modalities=["audio"],
    model_type=["dense"],
    extra_requirements_groups=["audio", "timm"],
)

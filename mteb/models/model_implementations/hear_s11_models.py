from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator, single_clip_dataloader

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType
    from mteb.types._encoder_io import AudioInput


class HeARS11AudioWrapper(AbsEncoder):
    sampling_rate = 16_000
    # 2 s: clip_seconds=2.0
    # https://huggingface.co/matthewagi/HeAR-s1.1/blob/main/config.json
    clip_samples = 32_000

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoModel

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

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
        """Split mono audio into 2 s windows (2 s hop, last one zero-padded), as in the author's MAEB run.

        https://huggingface.co/matthewagi/HeAR-s1.1/blob/main/maeb_run_config.json
        """
        import torch

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        audio = audio.reshape(-1)

        n_windows = max(1, math.ceil(audio.shape[0] / cls.clip_samples))
        audio = np.pad(audio, (0, n_windows * cls.clip_samples - audio.shape[0]))

        return torch.from_numpy(audio.reshape(n_windows, cls.clip_samples))

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        import torch

        embeddings = []

        inputs = single_clip_dataloader(
            inputs,
            "HeAR's model code changes a clip's embedding with its batch-mates, even at a fixed 2 s length",
        )
        for batch in tqdm(inputs, disable=not show_progress_bar):
            for item in batch["audio"]:
                windows = self._prepare_audio(item["array"]).to(self.device)
                with torch.no_grad():
                    # one window at a time, for the same batch dependence
                    window_embeddings = [
                        self.model(
                            input_values=window[None], return_dict=True
                        ).pooler_output
                        for window in windows
                    ]
                embeddings.append(
                    torch.cat(window_embeddings).mean(dim=0, keepdim=True).cpu()
                )

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
        inputs.collate_fn = AudioCollator(target_sampling_rate=self.sampling_rate)
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
    memory_usage_mb=84,
    embed_dim=384,
    license="https://developers.google.com/health-ai-developer-foundations/terms",
    reference="https://huggingface.co/matthewagi/HeAR-s1.1",
    similarity_fn_name="cosine",
    framework=["PyTorch", "Transformers", "safetensors"],
    use_instructions=False,
    public_training_code="https://github.com/Matthew-agi/hear-distillation-datalake",
    public_training_data="https://huggingface.co/datasets/laion/LAION-Audio-300M",
    training_datasets=set(),
    modalities=["audio"],
    model_type=["dense"],
    extra_requirements_groups=["audio", "timm"],
)

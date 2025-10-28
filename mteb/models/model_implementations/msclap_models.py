import logging
import warnings
from functools import partial
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb import TaskMetadata
from mteb._requires_package import requires_package
from mteb.models import ModelMeta
from mteb.types import Array, BatchedInput, PromptType
from mteb.types._encoder_io import AudioInput, TextInput

logger = logging.getLogger(__name__)


class MSClapWrapper:
    def __init__(
        self,
        model_name: str = "microsoft/msclap-2023",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_s: float = 30.0,
        **kwargs: Any,
    ):
        requires_package(
            self,
            "msclap",
            "pip install 'mteb[msclap]'",
        )
        from msclap import CLAP

        self.model_name = model_name
        self.device = device
        self.sampling_rate = 48000
        self.max_audio_length_s = max_audio_length_s

        if "2022" in self.model_name:
            self.version = "2022"
            self.text_length = 100
        elif "2023" in self.model_name:
            self.version = "2023"
            self.text_length = 77
        else:
            self.version = "2023"
            self.text_length = 77

        self.use_cuda = device == "cuda"
        self.model = CLAP(version=self.version, use_cuda=self.use_cuda)
        self.model.clap = self.model.clap.to(self.device)
        self.tokenizer = self.model.tokenizer

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        import torchaudio

        all_embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
        ):
            audio_arrays = []
            for a in batch["audio"]:
                array = torch.tensor(a["array"], dtype=torch.float32)
                sr = a.get("sampling_rate", None)
                if sr is None:
                    warnings.warn(
                        f"No sampling_rate provided for an audio sample. "
                        f"Assuming {self.sampling_rate} Hz (model default)."
                    )
                    sr = self.sampling_rate

                if sr != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=self.sampling_rate
                    )
                    array = resampler(array)
                audio_arrays.append(array.numpy())

            with torch.no_grad():
                # Use the internal audio encoder directly
                # [0] gives audio embeddings, [1] gives class probabilities
                audio_features = self.model.clap.audio_encoder(audio_arrays)[0]

                # Normalize embeddings
                audio_features = audio_features / audio_features.norm(
                    dim=-1, keepdim=True
                )
                all_embeddings.append(audio_features.cpu().numpy())

        return np.vstack(all_embeddings)

    def get_text_embeddings(
        self,
        inputs: DataLoader[TextInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        text_embeddings = []
        for batch in tqdm(
            inputs, disable=not show_progress_bar, desc="Processing text batches"
        ):
            texts = batch["text"]

            features = self.tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.text_length,
            )
            features = {k: v.to(self.device) for k, v in features.items()}

            with torch.no_grad():
                text_features = self.model.clap.caption_encoder(features)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_embeddings.append(text_features.cpu().numpy())

        return np.vstack(text_embeddings)

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
        text_embeddings = None
        audio_embeddings = None

        if "text" in inputs.dataset.features:
            text_embeddings = self.get_text_embeddings(inputs, **kwargs)
        if "audio" in inputs.dataset.features:
            audio_embeddings = self.get_audio_embeddings(inputs, **kwargs)

        if text_embeddings is not None and audio_embeddings is not None:
            if len(text_embeddings) != len(audio_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            fused_embeddings = text_embeddings + audio_embeddings
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif audio_embeddings is not None:
            return audio_embeddings
        raise ValueError


# Microsoft CLAP Model metadata
ms_clap_2022 = ModelMeta(
    loader=partial(MSClapWrapper, model_name="microsoft/msclap-2022"),
    name="microsoft/msclap-2022",
    languages=["eng-Latn"],
    revision="N/A",
    release_date="2022-12-01",
    modalities=["audio", "text"],
    n_parameters=196_000_000,
    memory_usage_mb=750,
    max_tokens=None,
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/microsoft/CLAP",
    public_training_data="https://github.com/microsoft/CLAP",
    framework=["PyTorch"],
    reference="https://github.com/microsoft/CLAP",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=set(),
)

ms_clap_2023 = ModelMeta(
    loader=partial(MSClapWrapper, model_name="microsoft/msclap-2023"),
    name="microsoft/msclap-2023",
    languages=["eng-Latn"],
    revision="N/A",
    release_date="2023-09-01",
    modalities=["audio", "text"],
    n_parameters=160_000_000,
    memory_usage_mb=610,
    max_tokens=None,
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/microsoft/CLAP",
    public_training_data="https://github.com/microsoft/CLAP",
    framework=["PyTorch"],
    reference="https://github.com/microsoft/CLAP",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=set(),
)

import warnings
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb import TaskMetadata
from mteb._requires_package import requires_package
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, BatchedInput, PromptType
from mteb.types._encoder_io import AudioInput, TextInput


class MuQMuLanWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str = "OpenMuQ/MuQ-MuLan-large",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_s: float = 30.0,
        **kwargs: Any,
    ):
        requires_package(self, "muq", "pip install 'mteb[muq]'")
        from muq import MuQMuLan

        self.model_name = model_name
        self.device = device
        self.sampling_rate = 24000
        self.max_audio_length_s = max_audio_length_s
        # Apply audio truncation (30 seconds max)
        self.max_length_samples = int(self.max_audio_length_s * self.sampling_rate)

        # Load the model
        self.model = MuQMuLan.from_pretrained(model_name).eval().to(self.device)
        self.model.eval()

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        import torchaudio

        all_features = []

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
                # Apply audio truncation (30 seconds max)
                if array.shape[-1] > self.max_length_samples:
                    array = array[..., : self.max_length_samples]
                audio_arrays.append(array.numpy())

            # Find max length and pad all tensors
            max_length = max(tensor.shape[-1] for tensor in batch)
            batch_tensor = torch.zeros(len(batch), max_length, dtype=torch.float32)

            for idx, tensor in enumerate(batch):
                length = tensor.shape[-1]
                batch_tensor[idx, :length] = tensor

            batch_tensor = batch_tensor.to(self.device)

            with torch.no_grad():
                # Process entire batch at once
                audio_embeds = self.model(wavs=batch_tensor)
                all_features.extend(
                    [
                        embed.cpu().detach().numpy().reshape(1, -1)
                        for embed in audio_embeds
                    ]
                )

        return np.vstack(all_features)

    def get_text_embeddings(
        self,
        inputs: DataLoader[TextInput],
        **kwargs: Any,
    ) -> np.ndarray:
        """Get text embeddings using MuQ-MuLan."""
        texts = [text for batch in inputs for text in batch["text"]]
        text_embeds = self.model(texts=texts)

        return text_embeds.cpu().detach().numpy()

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

    def similarity(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Calculate similarity between audio and text embeddings."""
        embeddings1 = torch.from_numpy(embeddings1).to(self.device)
        embeddings2 = torch.from_numpy(embeddings2).to(self.device)

        with torch.no_grad():
            similarity = self.model.calc_similarity(embeddings1, embeddings2)

        return similarity.cpu().detach().numpy()


muq_mulan_large = ModelMeta(
    loader=MuQMuLanWrapper,
    name="OpenMuQ/MuQ-MuLan-large",
    languages=["eng-Latn", "zho-Hans"],  # English and Chinese support
    revision="8a081dbcf84edd47ea7db3c4ecb8fd1ec1ddacfe8a081dbcf84edd47ea7db3c4ecb8fd1ec1ddacfe",
    release_date="2025-01-01",
    modalities=["audio", "text"],
    n_parameters=630_000_000,
    memory_usage_mb=2530,
    max_tokens=None,
    embed_dim=512,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="https://github.com/tencent-ailab/MuQ",
    public_training_data="https://github.com/tencent-ailab/MuQ",
    framework=["PyTorch"],
    reference="https://huggingface.co/OpenMuQ/MuQ-MuLan-large",
    # https://github.com/tencent-ailab/MuQ/blob/28847ea50cd31ac4b8b6a7dacc051ad7d1c7606a/src/muq/muq_mulan/muq_mulan.py#L171
    similarity_fn_name="dot",
    use_instructions=False,
    training_datasets=set(),
)

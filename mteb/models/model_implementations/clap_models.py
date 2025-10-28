import warnings
from functools import partial
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

from mteb import TaskMetadata
from mteb.models import ModelMeta
from mteb.types import Array, BatchedInput, PromptType
from mteb.types._encoder_io import AudioInput, TextInput


class ClapZeroShotWrapper:
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.model = ClapModel.from_pretrained(model_name, revision=revision).to(
            self.device
        )
        self.processor = ClapProcessor.from_pretrained(model_name, revision=revision)
        # CLAP's expected sampling rate. If the input audio is not sampled at this rate,
        # it will raise a ValueError similar to: ValueError: The model corresponding to
        # this feature extractor: ClapFeatureExtractor was trained using a sampling rate
        # of 48000. Please make sure that the provided `raw_speech` input was sampled
        # with 48000 and not 44100.
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

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
                audio_arrays.append(array.numpy())

            features = self.processor(
                audios=audio_arrays,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            features = {k: v.to(self.device) for k, v in features.items()}

            with torch.no_grad():
                audio_features = self.model.get_audio_features(**features)
                audio_features = audio_features / audio_features.norm(
                    dim=-1, keepdim=True
                )
                all_features.append(audio_features.cpu().detach().numpy())

        return np.vstack(all_features)

    def get_text_embeddings(
        self,
        inputs: DataLoader[TextInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        text_embeddings = []
        for batch in tqdm(
            inputs, disable=not show_progress_bar, desc="Processing text batches"
        ):
            texts = batch["text"]
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            text_features = self.model.get_text_features(**inputs)
            # Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_embeddings.append(text_features.cpu().detach().numpy())

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


# Model metadata
clap_htsat_fused = ModelMeta(
    loader=ClapZeroShotWrapper,
    name="laion/clap-htsat-fused",
    languages=["eng-Latn"],
    revision="cca9e288ab447cee67d9ada1f85ddb46500f1401",
    release_date="2023-05-22",
    modalities=["audio", "text"],
    n_parameters=153_507_530,  # Calculated using torch.numel(model.parameters())
    memory_usage_mb=586,  # Calculated using model.calculate_memory_usage_mb()
    max_tokens=float("inf"),
    embed_dim=512,  # The project_dim in config.json is 512
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/LAION-AI/CLAP",
    public_training_data="https://laion.ai/blog/laion-audio-630k/",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/clap_htsat_fused",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=set(
        # "LAION-Audio-630K": ["https://laion.ai/blog/laion-audio-630k/"]
    ),
)


clap_htsat_unfused = ModelMeta(
    loader=ClapZeroShotWrapper,
    name="laion/clap-htsat-unfused",
    languages=["eng-Latn"],
    revision="8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
    release_date="2023-05-22",
    modalities=["audio", "text"],
    n_parameters=153_492_890,  # Calculated using torch.numel(model.parameters())
    memory_usage_mb=586,  # Calculated using model.calculate_memory_usage_mb()
    max_tokens=float("inf"),
    embed_dim=512,  # The project_dim in config.json is 512
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/LAION-AI/CLAP",
    public_training_data="https://laion.ai/blog/laion-audio-630k/",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/clap_htsat_unfused",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=set(
        # "LAION-Audio-630K": ["https://laion.ai/blog/laion-audio-630k/"]
    ),
)

larger_clap_general = ModelMeta(
    loader=ClapZeroShotWrapper,
    name="laion/larger_clap_general",
    languages=["eng-Latn"],
    revision="ada0c23a36c4e8582805bb38fec3905903f18b41",
    release_date="2023-05-22",
    modalities=["audio", "text"],
    n_parameters=193_913_882,  # Calculated using torch.numel(model.parameters())
    memory_usage_mb=740,  # Calculated using model.calculate_memory_usage_mb()
    max_tokens=float("inf"),
    embed_dim=512,  # The project_dim (for even larger clap general) in config.json is 512
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/LAION-AI/CLAP",
    public_training_data="https://laion.ai/blog/laion-audio-630k/",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/larger_clap_general",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=set(
        # "LAION-Audio-630K": ["https://laion.ai/blog/laion-audio-630k/"]
    ),  # Additional finetuning over music dataset but not specified what the exact dataset is
)

larger_clap_music = ModelMeta(
    loader=ClapZeroShotWrapper,
    name="laion/larger_clap_music",
    languages=["eng-Latn"],
    revision="a0b4534a14f58e20944452dff00a22a06ce629d1",
    release_date="2023-05-22",
    modalities=["audio", "text"],
    n_parameters=193_913_882,  # Calculated using torch.numel(model.parameters())
    memory_usage_mb=740,  # Calculated using model.calculate_memory_usage_mb()
    max_tokens=float("inf"),
    embed_dim=512,  # The project_dim (for even larger clap general) in config.json is 512
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/LAION-AI/CLAP",
    public_training_data="https://laion.ai/blog/laion-audio-630k/",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/larger_clap_music",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=set(
        # "LAION-Audio-630K": ["https://laion.ai/blog/laion-audio-630k/"]
    ),  # Additional finetuning over music dataset but not specified what the exact dataset is
)

larger_clap_music_and_speech = ModelMeta(
    loader=partial(
        ClapZeroShotWrapper, model_name="laion/larger_clap_music_and_speech"
    ),
    name="laion/larger_clap_music_and_speech",
    languages=["eng-Latn"],
    revision="195c3a3e68faebb3e2088b9a79e79b43ddbda76b",
    release_date="2023-05-22",
    modalities=["audio", "text"],
    n_parameters=193_913_882,  # Calculated using torch.numel(model.parameters())
    memory_usage_mb=740,  # Calculated using model.calculate_memory_usage_mb()
    max_tokens=float("inf"),
    embed_dim=512,  # The project_dim (for even larger clap general) in config.json is 512
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/LAION-AI/CLAP",
    public_training_data="https://laion.ai/blog/laion-audio-630k/",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/larger_clap_music_and_speech",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=set(
        # "LAION-Audio-630K": ["https://laion.ai/blog/laion-audio-630k/"]
    ),  # Additional finetuning over music dataset but not specified what the exact dataset is
)

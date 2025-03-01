from __future__ import annotations

from functools import partial

import numpy as np
import torch
from datasets import Audio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

from mteb.encoder_interface import AudioEncoder, PromptType
from mteb.model_meta import ModelMeta


class WavlmWrapper(AudioEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.model_name = model_name
        self.model_revision = revision

        self.model = WavLMModel.from_pretrained(
            self.model_name, revision=self.model_revision
        )
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name, revision=self.model_revision
        )
        self.embed_dim = self.model.config.hidden_size

        if device:
            self.model = self.model.to(device)
        print("WavLM initialized.")

    def get_audio_embeddings(
        self, audio_files: list[Audio] | Audio, batch_size: int = 32, **kwargs
    ) -> np.ndarray:
        if not isinstance(audio_files, list):
            audio_files = [audio_files]

        all_embeddings = []

        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i : i + batch_size]

            audio_data = [file["array"] for file in batch]
            sampling_rates = [file["sampling_rate"] for file in batch]

            # Preprocess batch
            inputs = self.feature_extractor(
                audio_data,
                sampling_rate=sampling_rates[0],
                padding=True,
                return_tensors="pt",
            )

            if hasattr(self, "device") and self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(
                    input_values=inputs["input_values"],
                    output_hidden_states=True,
                    return_dict=True,
                )

            hidden_states = outputs.hidden_states[-1]
            batch_embeddings = hidden_states.mean(dim=1).cpu().numpy()
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def encode(
        self,
        audio_files: list[Audio],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        return self.get_audio_embeddings(audio_files, **kwargs)


wavlm_base = ModelMeta(
    loader=partial(WavlmWrapper, model_name="microsoft/wavlm-base"),
    name="microsoft/wavlm-base",
    languages=["eng"],
    open_weights=True,
    revision="main",
    release_date="2022-07-19",
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="MIT",
    reference="https://huggingface.co/microsoft/wavlm-base",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

wavlm_base_sd = ModelMeta(
    loader=partial(WavlmWrapper, model_name="microsoft/wavlm-base-sd"),
    name="microsoft/wavlm-base-sd",
    languages=["eng"],
    open_weights=True,
    revision="main",
    release_date="2022-07-19",
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="MIT",
    reference="https://huggingface.co/microsoft/wavlm-base-sd",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)
# print(f"wavlm_base: {wavlm_base.calculate_memory_usage_mb()}")

wavlm_base_plus = ModelMeta(
    loader=partial(WavlmWrapper, model_name="microsoft/wavlm-base-plus"),
    name="microsoft/wavlm-base-plus",
    languages=["eng"],
    open_weights=True,
    revision="main",
    release_date="2022-07-19",
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="MIT",
    reference="https://huggingface.co/microsoft/wavlm-base-plus",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

# print(f"wavlm_base_plus: {wavlm_base_plus.calculate_memory_usage_mb()}")

wavlm_base_plus_sv = ModelMeta(
    loader=partial(WavlmWrapper, model_name="microsoft/wavlm-base-plus-sv"),
    name="microsoft/wavlm-base-plus-sv",
    languages=["eng"],
    open_weights=True,
    revision="main",
    release_date="2022-07-19",  # estimate
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="MIT",
    reference="https://huggingface.co/microsoft/wavlm-base-plus-sv",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

wavlm_base_plus_sd = ModelMeta(
    loader=partial(WavlmWrapper, model_name="microsoft/wavlm-base-plus-sd"),
    name="microsoft/wavlm-base-plus-sd",
    languages=["eng"],
    open_weights=True,
    revision="main",
    release_date="2022-07-19",  # estimate
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="MIT",
    reference="https://huggingface.co/microsoft/wavlm-base-plus-sd",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

# print(f"wavlm_base_plus_sv: {wavlm_base_plus_sv.calculate_memory_usage_mb()}")

wavlm_base_sv = ModelMeta(
    loader=partial(WavlmWrapper, model_name="microsoft/wavlm-base-sv"),
    name="microsoft/wavlm-base-sv",
    languages=["eng"],
    open_weights=True,
    revision="main",
    release_date="2022-07-19",  # estimate
    max_tokens=float("inf"),
    n_parameters=94_700_000,
    memory_usage_mb=361,
    embed_dim=768,
    license="MIT",
    reference="https://huggingface.co/microsoft/wavlm-base-sv",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

# print(f"wavlm_base_sv: {wavlm_base_sv.calculate_memory_usage_mb()}")

wavlm_large = ModelMeta(
    loader=partial(WavlmWrapper, model_name="microsoft/wavlm-large"),
    name="microsoft/wavlm-large",
    languages=["eng"],
    open_weights=True,
    revision="main",
    release_date="2022-07-19",  # estimate
    max_tokens=float("inf"),
    n_parameters=316_620_000,
    memory_usage_mb=1208,
    embed_dim=1024,
    license="MIT",
    reference="https://huggingface.co/microsoft/wavlm-large",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

# print(f"wavlm_large: {wavlm_large.calculate_memory_usage_mb()}")

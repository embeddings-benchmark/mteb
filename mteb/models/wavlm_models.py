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
        model_revision: str,
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.model_name = model_name
        self.model_revision = model_revision

        self.model = WavLMModel.from_pretrained(
            self.model_name, revision=self.model_revision
        )
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name,
            # revision=self.model_revision
        )
        self.embed_dim = self.model.config.hidden_size

        if device:
            self.model = self.model.to(device)

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
        batch_size: int = 32,
        **kwargs,
    ) -> np.ndarray:
        return self.get_audio_embeddings(audio_files, batch_size=batch_size, **kwargs)


wavlm_base = ModelMeta(
    loader=partial(
        WavlmWrapper,
        model_name="microsoft/wavlm-base",
        model_revision="efa81aae7ff777e464159e0f877d54eac5b84f81",
    ),
    name="microsoft/wavlm-base",
    languages=["eng"],
    open_weights=True,
    revision="efa81aae7ff777e464159e0f877d54eac5b84f81",
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
    loader=partial(
        WavlmWrapper,
        model_name="microsoft/wavlm-base-sd",
        model_revision="fe13cca7e592cf0e11287cfede24e6999ac7dc4e",
    ),
    name="microsoft/wavlm-base-sd",
    languages=["eng"],
    open_weights=True,
    revision="fe13cca7e592cf0e11287cfede24e6999ac7dc4e",
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

wavlm_base_plus = ModelMeta(
    loader=partial(
        WavlmWrapper,
        model_name="microsoft/wavlm-base-plus",
        model_revision="4c66d4806a428f2e922ccfa1a962776e232d487b",
    ),
    name="microsoft/wavlm-base-plus",
    languages=["eng"],
    open_weights=True,
    revision="4c66d4806a428f2e922ccfa1a962776e232d487b",
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

wavlm_base_plus_sv = ModelMeta(
    loader=partial(
        WavlmWrapper,
        model_name="microsoft/wavlm-base-plus-sv",
        model_revision="feb593a6c23c1cc3d9510425c29b0a14d2b07b1e",
    ),
    name="microsoft/wavlm-base-plus-sv",
    languages=["eng"],
    open_weights=True,
    revision="feb593a6c23c1cc3d9510425c29b0a14d2b07b1e",
    release_date="2022-07-19",
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
    loader=partial(
        WavlmWrapper,
        model_name="microsoft/wavlm-base-plus-sd",
        model_revision="5bd86f0662bd55704109a794c6a1b1790ea0f91a",
    ),
    name="microsoft/wavlm-base-plus-sd",
    languages=["eng"],
    open_weights=True,
    revision="5bd86f0662bd55704109a794c6a1b1790ea0f91a",
    release_date="2022-07-19",
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


wavlm_base_sv = ModelMeta(
    loader=partial(
        WavlmWrapper,
        model_name="microsoft/wavlm-base-sv",
        model_revision="0a23162ffc49adcf42bdf836a00cb2eb45af3601",
    ),
    name="microsoft/wavlm-base-sv",
    languages=["eng"],
    open_weights=True,
    revision="0a23162ffc49adcf42bdf836a00cb2eb45af3601",
    release_date="2022-07-19",
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


wavlm_large = ModelMeta(
    loader=partial(
        WavlmWrapper,
        model_name="microsoft/wavlm-large",
        model_revision="c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
    ),
    name="microsoft/wavlm-large",
    languages=["eng"],
    open_weights=True,
    revision="c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
    release_date="2022-07-19",
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

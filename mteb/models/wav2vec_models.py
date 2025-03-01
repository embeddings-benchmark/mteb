from __future__ import annotations

from functools import partial

import numpy as np
import torch
from datasets import Audio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from mteb.encoder_interface import AudioEncoder, PromptType
from mteb.model_meta import ModelMeta


class Wav2vec2Wrapper(AudioEncoder):
    def __init__(
        self,
        device: str | None = None,
        model_name="facebook/wav2vec2-base",
        model_revision=None,
        **kwargs,
    ):
        self.device = device
        self.model = Wav2Vec2Model.from_pretrained(model_name, revision=model_revision)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, revision=model_revision
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

            if self.device:
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


wav2vec2_base = ModelMeta(
    loader=partial(
        Wav2vec2Wrapper,
        model_name="facebook/wav2vec2-base",
        model_revision="0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8",
    ),
    name="facebook/wav2vec2-base",
    languages=["en"],
    open_weights=True,
    revision="0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8",
    release_date="2020-10-26",
    max_tokens=float("inf"),
    n_parameters=95_000_000,
    memory_usage_mb=362,
    embed_dim=768,
    license="Apache-2.0",
    reference="https://huggingface.co/facebook/wav2vec2-base",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)


wav2vec2_base_960h = ModelMeta(
    loader=partial(
        Wav2vec2Wrapper,
        model_name="facebook/wav2vec2-base-960h",
        model_revision="22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
    ),
    name="facebook/wav2vec2-base-960h",
    languages=["en"],
    open_weights=True,
    revision="22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
    release_date="2020-10-26",
    max_tokens=float("inf"),
    n_parameters=95_000_000,
    memory_usage_mb=360,
    embed_dim=768,
    license="Apache-2.0",
    reference="https://huggingface.co/facebook/wav2vec2-base-960h",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)


wav2vec2_large = ModelMeta(
    loader=partial(
        Wav2vec2Wrapper,
        model_name="facebook/wav2vec2-large",
        model_revision="312b2410566b698c7a649068d413b2067848bd75",
    ),
    name="facebook/wav2vec2-large",
    languages=["en"],
    open_weights=True,
    revision="312b2410566b698c7a649068d413b2067848bd75",
    release_date="2020-10-26",
    max_tokens=float("inf"),
    n_parameters=317_000_000,
    memory_usage_mb=1_209,
    embed_dim=1_024,
    license="Apache-2.0",
    reference="https://huggingface.co/facebook/wav2vec2-large",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)


wav2vec2_large_xlsr_53 = ModelMeta(
    loader=partial(
        Wav2vec2Wrapper,
        model_name="facebook/wav2vec2-large-xlsr-53",
        model_revision="c3f9d884181a224a6ac87bf8885c84d1cff3384f",
    ),
    name="facebook/wav2vec2-large-xlsr-53",
    languages=["en"],
    open_weights=True,
    revision="c3f9d884181a224a6ac87bf8885c84d1cff3384f",
    release_date="2020-10-26",
    max_tokens=float("inf"),
    n_parameters=317_000_000,
    memory_usage_mb=1_209,
    embed_dim=1_024,
    license="Apache-2.0",
    reference="https://huggingface.co/facebook/wav2vec2-large-xlsr-53",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)


wav2vec2_lv_60_espeak_cv_ft = ModelMeta(
    loader=partial(
        Wav2vec2Wrapper,
        model_name="facebook/wav2vec2-lv-60-espeak-cv-ft",
        model_revision="ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4",
    ),
    name="facebook/wav2vec2-lv-60-espeak-cv-ft",
    languages=["en"],
    open_weights=True,
    revision="ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4",
    release_date="2020-10-26",
    max_tokens=float("inf"),
    n_parameters=317_000_000,
    memory_usage_mb=1_209,
    embed_dim=1_024,
    license="Apache-2.0",
    reference="https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

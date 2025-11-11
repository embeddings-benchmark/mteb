import logging
import warnings
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from mteb import TaskMetadata
from mteb._requires_package import requires_audio_dependencies
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, BatchedInput, PromptType
from mteb.types._encoder_io import AudioInput

logger = logging.getLogger(__name__)


class Qwen2AudioWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_seconds: float = 30.0,
        **kwargs: Any,
    ):
        requires_audio_dependencies()
        self.model_name = model_name
        self.device = device
        self.max_audio_length_seconds = max_audio_length_seconds

        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name, revision=revision
        ).to(self.device)
        self.model.eval()

        self.audio_encoder = self.model.audio_tower

        cfg = self.model.config.audio_config
        self.embed_dim = getattr(cfg, "d_model", getattr(cfg, "hidden_size", None))
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
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

            # Qwen2Audio specific: create prompt with <|AUDIO|> tokens
            prompt = " ".join(["<|AUDIO|>"] * len(audio_arrays))

            processor_inputs = self.processor(
                text=prompt,
                audio=audio_arrays,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.max_audio_length_seconds * self.sampling_rate),
            )

            input_features = processor_inputs.input_features.to(self.device)

            with torch.no_grad():
                outputs = self.audio_encoder(
                    input_features=input_features,
                    output_hidden_states=True,
                )

                # Use last hidden state and mean pool
                last_hidden = outputs.hidden_states[-1]
                embeddings = last_hidden.mean(dim=1).cpu().detach()
                all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0).numpy()

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
        if "audio" not in inputs.dataset.features:
            raise ValueError("Qwen2AudioWrapper only supports audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


qwen2_audio_meta = ModelMeta(
    loader=Qwen2AudioWrapper,
    name="Qwen/Qwen2-Audio-7B",
    languages=["eng-Latn"],
    open_weights=True,
    revision="dd84470756e6277a71d4d7188773a43cde92696e",
    release_date="2024-08-09",
    max_tokens=131_072,
    n_parameters=7_000_000_000,
    memory_usage_mb=None,
    embed_dim=1280,
    license="mit",
    reference="https://huggingface.co/Qwen/Qwen2-Audio-7B",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)

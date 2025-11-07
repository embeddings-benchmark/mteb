import logging
import warnings
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb import TaskMetadata
from mteb._requires_package import requires_audio_dependencies
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, BatchedInput, PromptType
from mteb.types._encoder_io import AudioInput

logger = logging.getLogger(__name__)


class ASTWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        requires_audio_dependencies()
        from transformers import ASTFeatureExtractor, ASTModel

        self.model_name = model_name
        self.device = device

        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name, revision=revision).to(
            self.device
        )
        self.model.eval()
        self.sampling_rate = self.feature_extractor.sampling_rate

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

            features = self.feature_extractor(
                audio_arrays,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).to(self.device)

            outputs = self.model(**features)

            # AST's pooled output is the [CLS] token embedding
            embeddings = outputs.pooler_output
            all_embeddings.append(embeddings.cpu().detach())
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
            raise ValueError("ASTWrapper only supports audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


# Model metadata
ast_audioset = ModelMeta(
    loader=ASTWrapper,
    name="MIT/ast-finetuned-audioset-10-10-0.4593",
    languages=["eng-Latn"],
    open_weights=True,
    revision="f826b80d28226b62986cc218e5cec390b1096902",
    release_date="2021-07-08",
    max_tokens=None,
    n_parameters=86_600_000,
    memory_usage_mb=330,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/YuanGongND/ast",
    public_training_data="https://research.google.com/audioset/dataset/index.html",
    training_datasets=set(),  # "AudioSet": ["train"]},
    modalities=["audio"],
)

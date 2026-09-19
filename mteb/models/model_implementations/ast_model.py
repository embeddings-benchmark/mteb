from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import ASTFeatureExtractor, ASTModel

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
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
        self.model_name = model_name
        self.device = device

        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name, revision=revision).to(
            self.device
        )
        self.model.eval()
        self.sampling_rate = self.feature_extractor.sampling_rate

    @torch.no_grad()
    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        inputs.collate_fn = AudioCollator(target_sampling_rate=self.sampling_rate)
        all_embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
        ):
            # the extractor always emits max_length=1024 frames (10.24 s)
            # https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/blob/main/preprocessor_config.json
            # pad up to the 400-sample FFT window so short clips do not crash
            audio_arrays = [
                np.pad(
                    np.asarray(a["array"]),
                    (0, max(0, 401 - np.asarray(a["array"]).shape[-1])),
                )
                for a in batch["audio"]
            ]

            features = self.feature_extractor(
                audio_arrays,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
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
    n_embedding_parameters=0,
    memory_usage_mb=330,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/YuanGongND/ast",
    public_training_data="https://research.google.com/audioset/dataset/index.html",
    training_datasets={
        "AudioSetMini",
    },
    modalities=["audio"],
    citation="""
@misc{gong2021astaudiospectrogramtransformer,
      title={AST: Audio Spectrogram Transformer},
      author={Yuan Gong and Yu-An Chung and James Glass},
      year={2021},
      eprint={2104.01778},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2104.01778},
}""",
)

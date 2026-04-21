from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs

E5_OMNI_CITATION = """@article{chen2026e5omni,
  title={e5-omni: Explicit Cross-modal Alignment for Omni-modal Embeddings},
  author={Chen, Haonan and Gao, Sicheng and Radu, Timofte and Tetsuya, Sakai and Dou, Zhicheng},
  journal={arXiv preprint arXiv:2601.03666},
  year={2026}
}"""


class E5OmniEncoder(AbsEncoder):
    """MTEB-compatible encoder for e5-omni (text / image / audio / video)."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        self.model.eval()

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        ds_features = inputs.dataset.features

        active_cols = [
            col
            for col in ("image", "audio", "video", "text")
            if col in ds_features and inputs.dataset[0].get(col) is not None
        ]

        all_inputs = [
            {col: batch[col][i] for col in active_cols}
            for batch in inputs
            for i in range(len(batch[active_cols[0]]))
        ]

        if prompt_type == PromptType.query:
            return self.model.encode_query(all_inputs, **kwargs)
        return self.model.encode_document(all_inputs, **kwargs)


e5_omni_7b = ModelMeta(
    loader=E5OmniEncoder,
    name="Haon-Chen/e5-omni-7B",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="ffea4ae1382fc26dc9fc337a89ced3fab58e408b",
    release_date="2026-01-06",
    modalities=["text", "image", "audio", "video"],
    n_parameters=8_931_813_888,
    n_embedding_parameters=544_997_376,
    memory_usage_mb=17042,
    embed_dim=3584,
    license="mit",
    max_tokens=32768,
    reference="https://huggingface.co/Haon-Chen/e5-omni-7B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
    training_datasets=set(),
    public_training_code=None,
    public_training_data=None,
    citation=E5_OMNI_CITATION,
)

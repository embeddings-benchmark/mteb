"""Model definitions for cnmoro's static embedding models.

Proposed addition to `mteb/models/model_implementations/cnmoro_models.py` in the
embeddings-benchmark/mteb repository.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from mteb.types import Array

MODEL2VEC_CITATION = """@software{minishlab2024model2vec,
  author       = {Stephan Tulkens and {van Dongen}, Thomas},
  title        = {Model2Vec: Fast State-of-the-Art Static Embeddings},
  year         = {2024},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17270888},
  url          = {https://github.com/MinishLab/model2vec},
  license      = {MIT}
}"""


class Model2VecStaticModelWrapper(AbsEncoder):
    """Loads a Model2Vec static model via `model2vec.StaticModel`.

    Needed for checkpoints that use vocabulary quantization (a `mapping` tensor
    projecting the tokenizer vocabulary onto a smaller set of shared embedding rows),
    which `sentence_transformers.models.StaticEmbedding` cannot load.
    """

    def __init__(self, model: str, revision: str | None = None, **kwargs: Any) -> None:
        from model2vec import StaticModel

        self.model = StaticModel.from_pretrained(model)

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        prompt_type: Any = None,
        **kwargs: Any,
    ) -> Array:
        texts = [text for batch in inputs for text in batch["text"]]
        embeddings = self.model.encode(
            texts, show_progress_bar=kwargs.get("show_progress_bar", False)
        )
        return np.asarray(embeddings, dtype=np.float32)


static_nomic_384_pten_v2 = ModelMeta(
    loader=Model2VecStaticModelWrapper,
    name="cnmoro/static-nomic-384-pten-v2",
    model_type=["dense"],
    languages=["eng-Latn", "por-Latn"],
    open_weights=True,
    revision="08981248ac665f19e71df149a2f2b5ef9b514bd7",
    release_date="2026-05-29",
    # 32000 x 384 embedding matrix + 276214 per-token scalar weights
    n_parameters=32000 * 384 + 276214,
    n_embedding_parameters=32000 * 384 + 276214,
    memory_usage_mb=50,
    max_tokens=np.inf,
    embed_dim=384,
    license="apache-2.0",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["NumPy", "Sentence Transformers", "safetensors"],
    reference="https://huggingface.co/cnmoro/static-nomic-384-pten-v2",
    use_instructions=False,
    adapted_from="nomic-ai/nomic-embed-text-v2-moe",
    superseded_by=None,
    # Distilled with Tokenlearn; finetuned on a pt-BR translation of MS MARCO triplets.
    training_datasets={"MSMARCO"},
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/cnmoro/AllTripletsMsMarco-PTBR",
    citation=MODEL2VEC_CITATION,
)

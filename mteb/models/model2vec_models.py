from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np

from mteb.model_meta import ModelMeta
from mteb.models.bge_models import bge_training_data
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)


class Model2VecWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        **kwargs,
    ) -> None:
        """Wrapper for Model2Vec models.

        Args:
            model_name: The Model2Vec model to load from HuggingFace Hub.
            **kwargs: Additional arguments to pass to the wrapper.
        """
        requires_package(self, "model2vec", model_name, "pip install 'mteb[model2vec]'")
        from model2vec import StaticModel  # type: ignore

        self.model_name = model_name
        self.static_model = StaticModel.from_pretrained(self.model_name)

    def encode(
        self,
        sentences: Sequence[str],
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        return self.static_model.encode(sentences).astype(np.float32)


m2v_base_glove_subword = ModelMeta(
    loader=partial(
        Model2VecWrapper,
        model_name="minishlab/M2V_base_glove_subword",
    ),
    name="minishlab/M2V_base_glove_subword",
    languages=["eng-Latn"],
    open_weights=True,
    revision="5f4f5ca159b7321a8b39739bba0794fa0debddf4",
    release_date="2024-09-21",
    n_parameters=int(103 * 1e6),
    memory_usage_mb=391,
    max_tokens=np.inf,  # Theoretically infinite
    embed_dim=256,
    license="mit",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/minishlab/M2V_base_glove_subword",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseded_by=None,
    training_datasets=bge_training_data,  # distilled
    public_training_code="https://github.com/MinishLab/model2vec",
    public_training_data=None,
)


m2v_base_glove = ModelMeta(
    loader=partial(
        Model2VecWrapper,
        model_name="minishlab/M2V_base_glove",
    ),
    name="minishlab/M2V_base_glove",
    languages=["eng-Latn"],
    open_weights=True,
    revision="38ebd7f10f71e67fa8db898290f92b82e9cfff2b",
    release_date="2024-09-21",
    n_parameters=int(102 * 1e6),
    memory_usage_mb=391,
    max_tokens=np.inf,
    embed_dim=256,
    license="mit",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/minishlab/M2V_base_glove",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseded_by=None,
    training_datasets=bge_training_data,  # distilled
    public_training_code="https://github.com/MinishLab/model2vec",
    public_training_data=None,
)

m2v_base_output = ModelMeta(
    loader=partial(
        Model2VecWrapper,
        model_name="minishlab/M2V_base_output",
    ),
    name="minishlab/M2V_base_output",
    languages=["eng-Latn"],
    open_weights=True,
    revision="02460ae401a22b09d2c6652e23371398329551e2",
    release_date="2024-09-21",
    n_parameters=int(7.56 * 1e6),
    memory_usage_mb=29,
    max_tokens=np.inf,
    embed_dim=256,
    license="mit",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/minishlab/M2V_base_output",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseded_by=None,
    training_datasets=bge_training_data,  # distilled
    public_training_code="https://github.com/MinishLab/model2vec",
    public_training_data=None,
)

m2v_multilingual_output = ModelMeta(
    loader=partial(
        Model2VecWrapper,
        model_name="minishlab/M2V_multilingual_output",
    ),
    name="minishlab/M2V_multilingual_output",
    languages=["eng-Latn"],
    open_weights=True,
    revision="2cf4ec4e1f51aeca6c55cf9b93097d00711a6305",
    release_date="2024-09-21",
    n_parameters=int(128 * 1e6),
    memory_usage_mb=489,
    max_tokens=np.inf,
    embed_dim=256,
    license="mit",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/minishlab/M2V_multilingual_output",
    use_instructions=False,
    adapted_from="sentence-transformers/LaBSE",
    superseded_by=None,
    training_datasets=None,
    public_training_code="https://github.com/MinishLab/model2vec",
    public_training_data=None,
)

potion_base_2m = ModelMeta(
    loader=partial(
        Model2VecWrapper,
        model_name="minishlab/potion-base-2M",
    ),
    name="minishlab/potion-base-2M",
    languages=["eng-Latn"],
    open_weights=True,
    revision="86db093558fbced2072b929eb1690bce5272bd4b",
    release_date="2024-10-29",
    n_parameters=2 * 1e6,
    memory_usage_mb=7,
    max_tokens=np.inf,
    embed_dim=64,
    license="mit",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/minishlab/potion-base-2M",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseded_by=None,
    training_datasets=bge_training_data,  # distilled
    public_training_code="https://github.com/MinishLab/model2vec",
    public_training_data=None,
)

potion_base_4m = ModelMeta(
    loader=partial(
        Model2VecWrapper,
        model_name="minishlab/potion-base-4M",
    ),
    name="minishlab/potion-base-4M",
    languages=["eng-Latn"],
    open_weights=True,
    revision="81b1802ada41afcd0987a37dc15e569c9fa76f04",
    release_date="2024-10-29",
    n_parameters=3.78 * 1e6,
    memory_usage_mb=14,
    max_tokens=np.inf,
    embed_dim=128,
    license="mit",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/minishlab/potion-base-4M",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseded_by=None,
    training_datasets=bge_training_data,  # distilled
    public_training_code="https://github.com/MinishLab/model2vec",
    public_training_data=None,
)

potion_base_8m = ModelMeta(
    loader=partial(
        Model2VecWrapper,
        model_name="minishlab/potion-base-8M",
    ),
    name="minishlab/potion-base-8M",
    languages=["eng-Latn"],
    open_weights=True,
    revision="dcbec7aa2d52fc76754ac6291803feedd8c619ce",
    release_date="2024-10-29",
    n_parameters=7.56 * 1e6,
    memory_usage_mb=29,
    max_tokens=np.inf,
    embed_dim=256,
    license="mit",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/minishlab/potion-base-8M",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseded_by=None,
    training_datasets=bge_training_data,  # distilled
    public_training_code="https://github.com/MinishLab/model2vec",
    public_training_data=None,
)

pubmed_bert_100k = ModelMeta(
    loader=partial(
        Model2VecWrapper, model_name="NeuML/pubmedbert-base-embeddings-100K"
    ),
    name="NeuML/pubmedbert-base-embeddings-100K",
    languages=["eng-Latn"],
    open_weights=True,
    revision="bac5e3b12fb8c650e92a19c41b436732c4f16e9e",
    release_date="2025-01-03",
    n_parameters=1 * 1e5,
    memory_usage_mb=0,
    max_tokens=np.inf,
    embed_dim=64,
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/NeuML/pubmedbert-base-embeddings-100K",
    use_instructions=False,
    adapted_from="NeuML/pubmedbert-base-embeddings",
    superseded_by=None,
    training_datasets={},
    public_training_code="https://huggingface.co/NeuML/pubmedbert-base-embeddings-100K#training",
    public_training_data="https://pubmed.ncbi.nlm.nih.gov/download/",
)

pubmed_bert_500k = ModelMeta(
    loader=partial(
        Model2VecWrapper, model_name="NeuML/pubmedbert-base-embeddings-500K"
    ),
    name="NeuML/pubmedbert-base-embeddings-500K",
    languages=["eng-Latn"],
    open_weights=True,
    revision="34ba71e35c393fdad7ed695113f653feb407b16b",
    release_date="2025-01-03",
    n_parameters=5 * 1e5,
    memory_usage_mb=2,
    max_tokens=np.inf,
    embed_dim=64,
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/NeuML/pubmedbert-base-embeddings-500K",
    use_instructions=False,
    adapted_from="NeuML/pubmedbert-base-embeddings",
    superseded_by=None,
    training_datasets={},
    public_training_code="https://huggingface.co/NeuML/pubmedbert-base-embeddings-500K#training",
    public_training_data="https://pubmed.ncbi.nlm.nih.gov/download/",
)

pubmed_bert_1m = ModelMeta(
    loader=partial(Model2VecWrapper, model_name="NeuML/pubmedbert-base-embeddings-1M"),
    name="NeuML/pubmedbert-base-embeddings-1M",
    languages=["eng-Latn"],
    open_weights=True,
    revision="2b7fed222594708da6d88bcda92ae9b434b7ddd1",
    release_date="2025-01-03",
    n_parameters=1 * 1e6,
    memory_usage_mb=2,
    max_tokens=np.inf,
    embed_dim=64,
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/NeuML/pubmedbert-base-embeddings-1M",
    use_instructions=False,
    adapted_from="NeuML/pubmedbert-base-embeddings",
    superseded_by=None,
    training_datasets={},
    public_training_code="https://huggingface.co/NeuML/pubmedbert-base-embeddings-1M#training",
    public_training_data="https://pubmed.ncbi.nlm.nih.gov/download/",
)

pubmed_bert_2m = ModelMeta(
    loader=partial(Model2VecWrapper, model_name="NeuML/pubmedbert-base-embeddings-2M"),
    name="NeuML/pubmedbert-base-embeddings-2M",
    languages=["eng-Latn"],
    open_weights=True,
    revision="1d7bbe04d6713e425161146bfdc71473cbed498a",
    release_date="2025-01-03",
    n_parameters=1.95 * 1e6,
    memory_usage_mb=7,
    max_tokens=np.inf,
    embed_dim=64,
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/NeuML/pubmedbert-base-embeddings-2M",
    use_instructions=False,
    adapted_from="NeuML/pubmedbert-base-embeddings",
    superseded_by=None,
    training_datasets={},
    public_training_code="https://huggingface.co/NeuML/pubmedbert-base-embeddings-2M#training",
    public_training_data="https://pubmed.ncbi.nlm.nih.gov/download/",
)

pubmed_bert_8m = ModelMeta(
    loader=partial(Model2VecWrapper, model_name="NeuML/pubmedbert-base-embeddings-8M"),
    name="NeuML/pubmedbert-base-embeddings-8M",
    languages=["eng-Latn"],
    open_weights=True,
    revision="387d350015e963744f4fafe56a574b7cd48646c9",
    release_date="2025-01-03",
    n_parameters=7.81 * 1e6,
    memory_usage_mb=30,
    max_tokens=np.inf,
    embed_dim=256,
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/NeuML/pubmedbert-base-embeddings-8M",
    use_instructions=False,
    adapted_from="NeuML/pubmedbert-base-embeddings",
    superseded_by=None,
    training_datasets={},
    public_training_code="https://huggingface.co/NeuML/pubmedbert-base-embeddings-8M#training",
    public_training_data="https://pubmed.ncbi.nlm.nih.gov/download/",
)

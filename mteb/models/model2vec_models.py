from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta

from .wrapper import Wrapper

logger = logging.getLogger(__name__)


class Model2VecWrapper(Wrapper):
    def __init__(
        self,
        model: str,
        **kwargs,
    ) -> None:
        """Wrapper for Model2Vec models.

        Args:
            model: The Model2Vec model to use. Can be a string (model name), a SentenceTransformer model, or a CrossEncoder model.
        """
        from model2vec import StaticModel

        self.model = model
        self.static_model = StaticModel.from_pretrained(self.model)

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
        return self.static_model.encode(sentences)


m2v_base_glove_subword = ModelMeta(
    loader=partial(
        Model2VecWrapper,
        model="minishlab/M2V_base_glove_subword",
    ),
    name="minishlab/M2V_base_glove_subword",
    languages=["eng_Latn"],
    open_weights=True,
    revision="5f4f5ca159b7321a8b39739bba0794fa0debddf4",
    release_date="2024-10-14",  # blog post date
    n_parameters=103 * 1e6,
    max_tokens=1000000,  # I took this from the current leaderboard, I'm not sure if it's unlimited in theory, tough
    embed_dim=256,
    license="mit",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    reference="https://huggingface.co/minishlab/M2V_base_glove_subword",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseeded_by=None,
)


m2v_base_glove = ModelMeta(
    loader=partial(
        Model2VecWrapper,
        model="minishlab/M2V_base_glove",
    ),
    name="minishlab/M2V_base_glove",
    languages=["eng_Latn"],
    open_weights=True,
    revision="38ebd7f10f71e67fa8db898290f92b82e9cfff2b",
    release_date="2024-10-14",  # blog post date
    n_parameters=102 * 1e6,
    max_tokens=1000000,
    embed_dim=256,
    license="mit",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    reference="https://huggingface.co/minishlab/M2V_base_glove",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseeded_by=None,
)

m2v_base_output = ModelMeta(
    loader=partial(
        Model2VecWrapper,
        model="minishlab/M2V_base_output",
    ),
    name="minishlab/M2V_base_output",
    languages=["eng_Latn"],
    open_weights=True,
    revision="02460ae401a22b09d2c6652e23371398329551e2",
    release_date="2024-10-14",  # blog post date
    n_parameters=7.56 * 1e6,
    max_tokens=1000000,
    embed_dim=256,
    license="mit",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    reference="https://huggingface.co/minishlab/M2V_base_output",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseeded_by=None,
)

m2v_multilingual_output = ModelMeta(
    loader=partial(
        Model2VecWrapper,
        model="minishlab/M2V_multilingual_output",
    ),
    name="minishlab/M2V_multilingual_output",
    languages=["eng_Latn"],
    open_weights=True,
    revision="2cf4ec4e1f51aeca6c55cf9b93097d00711a6305",
    release_date="2024-10-14",  # blog post date
    n_parameters=128 * 1e6,
    max_tokens=1000000,
    embed_dim=256,
    license="mit",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    reference="https://huggingface.co/minishlab/M2V_multilingual_output",
    use_instructions=False,
    adapted_from="sentence-transformers/LaBSE",
    superseeded_by=None,
)

potion_base_2m = ModelMeta(
    loader=partial(
        Model2VecWrapper,
        model="minishlab/potion-base-2M",
    ),
    name="minishlab/potion-base-2M",
    languages=["eng_Latn"],
    open_weights=True,
    revision="86db093558fbced2072b929eb1690bce5272bd4b",
    release_date="2024-10-14",  # blog post date
    n_parameters=2 * 1e6,
    max_tokens=1000000,
    embed_dim=256,
    license="mit",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    reference="https://huggingface.co/minishlab/potion-base-2M",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseeded_by=None,
)

potion_base_4m = ModelMeta(
    loader=partial(
        Model2VecWrapper,
        model="minishlab/potion-base-4M",
    ),
    name="minishlab/potion-base-4M",
    languages=["eng_Latn"],
    open_weights=True,
    revision="81b1802ada41afcd0987a37dc15e569c9fa76f04",
    release_date="2024-10-14",  # blog post date
    n_parameters=3.78 * 1e6,
    max_tokens=1000000,
    embed_dim=256,
    license="mit",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    reference="https://huggingface.co/minishlab/potion-base-4M",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseeded_by=None,
)

potion_base_8m = ModelMeta(
    loader=partial(
        Model2VecWrapper,
        model="minishlab/potion-base-8M",
    ),
    name="minishlab/potion-base-8M",
    languages=["eng_Latn"],
    open_weights=True,
    revision="dcbec7aa2d52fc76754ac6291803feedd8c619ce",
    release_date="2024-10-14",  # blog post date
    n_parameters=7.56 * 1e6,
    max_tokens=1000000,
    embed_dim=256,
    license="mit",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    reference="https://huggingface.co/minishlab/potion-base-8M",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseeded_by=None,
)

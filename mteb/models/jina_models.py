from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import torch
from sentence_transformers import __version__ as st_version

from mteb.model_meta import ModelMeta

from ..encoder_interface import PromptType
from .sentence_transformer_wrapper import SentenceTransformerWrapper

logger = logging.getLogger(__name__)

MIN_SENTENCE_TRANSFORMERS_VERSION = (3, 1, 0)
CURRENT_SENTENCE_TRANSFORMERS_VERSION = tuple(map(int, st_version.split(".")))

XLMR_LANGUAGES = [
    "afr_Latn",
    "amh_Latn",
    "ara_Latn",
    "asm_Latn",
    "aze_Latn",
    "bel_Latn",
    "bul_Latn",
    "ben_Latn",
    "ben_Beng",
    "bre_Latn",
    "bos_Latn",
    "cat_Latn",
    "ces_Latn",
    "cym_Latn",
    "dan_Latn",
    "deu_Latn",
    "ell_Latn",
    "eng_Latn",
    "epo_Latn",
    "spa_Latn",
    "est_Latn",
    "eus_Latn",
    "fas_Latn",
    "fin_Latn",
    "fra_Latn",
    "fry_Latn",
    "gle_Latn",
    "gla_Latn",
    "glg_Latn",
    "guj_Latn",
    "hau_Latn",
    "heb_Latn",
    "hin_Latn",
    "hin_Deva",
    "hrv_Latn",
    "hun_Latn",
    "hye_Latn",
    "ind_Latn",
    "isl_Latn",
    "ita_Latn",
    "jpn_Latn",
    "jav_Latn",
    "kat_Latn",
    "kaz_Latn",
    "khm_Latn",
    "kan_Latn",
    "kor_Latn",
    "kur_Latn",
    "kir_Latn",
    "lat_Latn",
    "lao_Latn",
    "lit_Latn",
    "lav_Latn",
    "mlg_Latn",
    "mkd_Latn",
    "mal_Latn",
    "mon_Latn",
    "mar_Latn",
    "msa_Latn",
    "mya_Latn",
    "nep_Latn",
    "nld_Latn",
    "nob_Latn",
    "orm_Latn",
    "ori_Latn",
    "pan_Latn",
    "pol_Latn",
    "pus_Latn",
    "por_Latn",
    "ron_Latn",
    "rus_Latn",
    "san_Latn",
    "snd_Latn",
    "sin_Latn",
    "slk_Latn",
    "slv_Latn",
    "som_Latn",
    "sqi_Latn",
    "srp_Latn",
    "sun_Latn",
    "swe_Latn",
    "swa_Latn",
    "tam_Latn",
    "tam_Taml",
    "tel_Latn",
    "tel_Telu",
    "tha_Latn",
    "tgl_Latn",
    "tur_Latn",
    "uig_Latn",
    "ukr_Latn",
    "urd_Latn",
    "urd_Arab",
    "uzb_Latn",
    "vie_Latn",
    "xho_Latn",
    "yid_Latn",
    "zho_Hant",
    "zho_Hans",
]


class JinaWrapper(SentenceTransformerWrapper):
    """following the hf model card documentation."""

    jina_task_to_prompt = {
        "retrieval.query": "Represent the query for retrieving evidence documents: ",
        "retrieval.passage": "Represent the document for retrieval: ",
    }

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        if CURRENT_SENTENCE_TRANSFORMERS_VERSION < MIN_SENTENCE_TRANSFORMERS_VERSION:
            raise RuntimeError(
                f"sentence_transformers version {st_version} is lower than the required version 3.1.0"
            )
        try:
            import einops  # noqa: F401
        except ImportError:
            raise ImportError(
                "To use the jina-embeddings-v3 models `einops` is required. Please install it with `pip install mteb[jina]`."
            )
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.warning(
                "Using flash_attn for jina-embeddings-v3 models is recommended. Please install it with `pip install mteb[flash_attention]`."
                "Fallback to native implementation."
            )
        super().__init__(model, revision, model_prompts, **kwargs)

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        prompt_name = self.get_prompt_name(self.model_prompts, task_name, prompt_type)
        if prompt_name:
            logger.info(
                f"Using prompt_name={prompt_name} for task={task_name} prompt_type={prompt_type}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_name} prompt_type={prompt_type}"
            )
        logger.info(f"Encoding {len(sentences)} sentences.")

        jina_task_name = self.model_prompts.get(prompt_name, None)

        embeddings = self.model.encode(
            sentences,
            task=jina_task_name,
            prompt=self.jina_task_to_prompt.get(jina_task_name, None),
            **kwargs,
        )

        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings


jina_embeddings_v3 = ModelMeta(
    loader=partial(  # type: ignore
        JinaWrapper,
        model="jinaai/jina-embeddings-v3",
        revision="215a6e121fa0183376388ac6b1ae230326bfeaed",
        trust_remote_code=True,
        model_prompts={
            "Retrieval-query": "retrieval.query",
            "Retrieval-passage": "retrieval.passage",
            "Clustering": "separation",
            "Classification": "classification",
            "STS": "text-matching",
            "PairClassification": "classification",
            "BitextMining": "text-matching",
            "MultilabelClassification": "classification",
            "Reranking": "separation",
            "Summarization": "text-matching",
        },
    ),
    name="jinaai/jina-embeddings-v3",
    languages=XLMR_LANGUAGES,
    open_weights=True,
    revision="215a6e121fa0183376388ac6b1ae230326bfeaed",
    release_date="2024-09-18",  # official release date
    n_parameters=int(572 * 1e6),
    max_tokens=8194,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    reference="https://huggingface.co/jinaai/jina-embeddings-v3",
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        # CulturaX
        "STS12": [],
        # "SICK": [],
        # "WMT19": [],
        # "MADLAD-3B": [],
        # NLI
        "MSMARCO": ["train"],
        "MSMARCOHardNegatives": ["train"],
        "NanoMSMARCORetrieval": ["train"],
        "NQ": ["train"],
        "NQHardNegatives": ["train"],
        "NanoNQRetrieval": ["train"],
        "NQ-PL": ["train"],  # translation not trained on
        # oasst1, oasst2
    },
    adapted_from="XLM-RoBERTa",
)


jina_embeddings_v2_base_en = ModelMeta(
    loader=partial(
        SentenceTransformerWrapper,
        model_name="jinaai/jina-embeddings-v2-base-en",
        revision="6e85f575bc273f1fd840a658067d0157933c83f0",
        trust_remote_code=True,
    ),
    name="jinaai/jina-embeddings-v2-base-en",
    languages=["eng-Latn"],
    open_weights=True,
    revision="6e85f575bc273f1fd840a658067d0157933c83f0",
    release_date="2023-09-27",
    n_parameters=137_000_000,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/jinaai/jina-embeddings-v2-base-en",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets=None,
    public_training_code=None,
    public_training_data=None,
)

jina_embeddings_v2_small_en = ModelMeta(
    loader=partial(
        SentenceTransformerWrapper,
        model_name="jinaai/jina-embeddings-v2-small-en",
        revision="796cff318cdd4e5fbe8b7303a1ef8cbec36996ef",
        trust_remote_code=True,
    ),
    name="jinaai/jina-embeddings-v2-small-en",
    languages=["eng-Latn"],
    open_weights=True,
    revision="796cff318cdd4e5fbe8b7303a1ef8cbec36996ef",
    release_date="2023-09-27",
    n_parameters=32_700_000,
    embed_dim=512,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/jinaai/jina-embeddings-v2-small-en",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets=None,
    public_training_code=None,
    public_training_data=None,
)

jina_embedding_b_en_v1 = ModelMeta(
    loader=partial(
        SentenceTransformerWrapper,
        model_name="jinaai/jina-embedding-b-en-v1",
        revision="aa0645035294a8c0607ce5bb700aba982cdff32c",
        trust_remote_code=True,
    ),
    name="jinaai/jina-embedding-b-en-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="aa0645035294a8c0607ce5bb700aba982cdff32c",
    release_date="2023-07-07",
    n_parameters=110_000_000,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/jinaai/jina-embedding-b-en-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by="jinaai/jina-embeddings-v2-base-en",
    adapted_from=None,
    training_datasets=None,
    public_training_code=None,
    public_training_data=None,
)

jina_embedding_s_en_v1 = ModelMeta(
    loader=partial(
        SentenceTransformerWrapper,
        model_name="jinaai/jina-embedding-s-en-v1",
        revision="c1fed70aa4823a640f1a7150a276e4d3b08dce08",
        trust_remote_code=True,
    ),
    name="jinaai/jina-embedding-s-en-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="c1fed70aa4823a640f1a7150a276e4d3b08dce08",
    release_date="2023-07-07",
    n_parameters=35_000_000,
    embed_dim=512,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/jinaai/jina-embedding-s-en-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by="jinaai/jina-embeddings-v2-small-en",
    adapted_from=None,
    training_datasets=None,
    public_training_code=None,
    public_training_data=None,
)

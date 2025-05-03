from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import torch
from sentence_transformers import __version__ as st_version

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)

MIN_SENTENCE_TRANSFORMERS_VERSION = (3, 1, 0)
CURRENT_SENTENCE_TRANSFORMERS_VERSION = tuple(map(int, st_version.split(".")))

XLMR_LANGUAGES = [
    "afr-Latn",
    "amh-Latn",
    "ara-Latn",
    "asm-Latn",
    "aze-Latn",
    "bel-Latn",
    "bul-Latn",
    "ben-Latn",
    "ben-Beng",
    "bre-Latn",
    "bos-Latn",
    "cat-Latn",
    "ces-Latn",
    "cym-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Latn",
    "eng-Latn",
    "epo-Latn",
    "spa-Latn",
    "est-Latn",
    "eus-Latn",
    "fas-Latn",
    "fin-Latn",
    "fra-Latn",
    "fry-Latn",
    "gle-Latn",
    "gla-Latn",
    "glg-Latn",
    "guj-Latn",
    "hau-Latn",
    "heb-Latn",
    "hin-Latn",
    "hin-Deva",
    "hrv-Latn",
    "hun-Latn",
    "hye-Latn",
    "ind-Latn",
    "isl-Latn",
    "ita-Latn",
    "jpn-Latn",
    "jav-Latn",
    "kat-Latn",
    "kaz-Latn",
    "khm-Latn",
    "kan-Latn",
    "kor-Latn",
    "kur-Latn",
    "kir-Latn",
    "lat-Latn",
    "lao-Latn",
    "lit-Latn",
    "lav-Latn",
    "mlg-Latn",
    "mkd-Latn",
    "mal-Latn",
    "mon-Latn",
    "mar-Latn",
    "msa-Latn",
    "mya-Latn",
    "nep-Latn",
    "nld-Latn",
    "nob-Latn",
    "orm-Latn",
    "ori-Latn",
    "pan-Latn",
    "pol-Latn",
    "pus-Latn",
    "por-Latn",
    "ron-Latn",
    "rus-Latn",
    "san-Latn",
    "snd-Latn",
    "sin-Latn",
    "slk-Latn",
    "slv-Latn",
    "som-Latn",
    "sqi-Latn",
    "srp-Latn",
    "sun-Latn",
    "swe-Latn",
    "swa-Latn",
    "tam-Latn",
    "tam-Taml",
    "tel-Latn",
    "tel-Telu",
    "tha-Latn",
    "tgl-Latn",
    "tur-Latn",
    "uig-Latn",
    "ukr-Latn",
    "urd-Latn",
    "urd-Arab",
    "uzb-Latn",
    "vie-Latn",
    "xho-Latn",
    "yid-Latn",
    "zho-Hant",
    "zho-Hans",
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
        requires_package(self, "einops", model, "pip install 'mteb[jina]'")
        import einops  # noqa: F401

        requires_package(
            self, "flash_attn", model, "pip install 'mteb[flash_attention]'"
        )
        import flash_attn  # noqa: F401

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
    memory_usage_mb=1092,
    max_tokens=8194,
    embed_dim=1024,
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
        "mMARCO-NL": ["train"],  # translation not trained on
        "NQ": ["train"],
        "NQHardNegatives": ["train"],
        "NanoNQRetrieval": ["train"],
        "NQ-PL": ["train"],  # translation not trained on
        "NQ-NL": ["train"],  # translation not trained on
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
    memory_usage_mb=262,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/jinaai/jina-embeddings-v2-base-en",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from="jina-bert-base-en-v1",  # pretrained on C4 with Alibi to support longer context.
    training_datasets={
        "PAQ": ["train"],
        "GooAQ": ["train"],
        "WikiAnswers": ["train"],
        "AmazonQA": ["train"],
        "ELI5": ["train"],
        "SentenceCompression": ["train"],
        "SimpleWikipedia": ["train"],
        "Specter": ["train"],
        "Squad2": ["train"],
        "Tmdb": ["train"],
        "TrivialQA": ["train"],
        "TweetQA": ["train"],
        "WikiHow": ["train"],
        "Xmarket": [],  # adopted from Cross-Market Recommendation (XMRec).
        "S2ORC": [],  # title abstract pair.
        "YahooAnswers": [],  # question answer pair.
        "MSMARCO": ["train"],  # pairs and mined hard negative.
        "StackExchange": [],  # title body pair.
        "QuoraQA": ["train"],  # duplicate question pairs.
        "MsCocoCaptions": ["train"],  # pairs describe the same image.
        "Flickr30k": ["train"],  # pairs describe the same image.
        "SNLI": ["train"],  # random negative.
        "ESCI": ["train"],  # exact match as positive match and mined hard negative.
        "NegationDataset": [
            "train"
        ],  # synthetically generated negation dataset https://huggingface.co/datasets/jinaai/negation-dataset
        "NQ": ["train"],  # mined hard negative.
        "HotpotQA": ["train"],  # mined hard negative.
        "FEVER": ["train"],  # mined hard negative.
        "CC-NEWS": [],  # title-content with random negative.
    },
    public_training_code=None,
    public_training_data=None,
)

jina_embeddings_v2_small_en = ModelMeta(
    loader=partial(
        SentenceTransformerWrapper,
        model_name="jinaai/jina-embeddings-v2-small-en",
        revision="44e7d1d6caec8c883c2d4b207588504d519788d0",
        trust_remote_code=True,
    ),
    name="jinaai/jina-embeddings-v2-small-en",
    languages=["eng-Latn"],
    open_weights=True,
    revision="44e7d1d6caec8c883c2d4b207588504d519788d0",
    release_date="2023-09-27",
    n_parameters=32_700_000,
    memory_usage_mb=62,
    embed_dim=512,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/jinaai/jina-embeddings-v2-small-en",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from="jina-bert-smalll-en-v1",  # pretrained on C4 with Alibi to support longer context
    training_datasets={
        "PAQ": ["train"],
        "GooAQ": ["train"],
        "WikiAnswers": ["train"],
        "AmazonQA": ["train"],
        "ELI5": ["train"],
        "SentenceCompression": ["train"],
        "SimpleWikipedia": ["train"],
        "Specter": ["train"],
        "Squad2": ["train"],
        "Tmdb": ["train"],
        "TrivialQA": ["train"],
        "TweetQA": ["train"],
        "WikiHow": ["train"],
        "Xmarket": [],  # adopted from Cross-Market Recommendation (XMRec).
        "S2ORC": [],  # title abstract pair.
        "YahooAnswers": [],  # question answer pair.
        "MSMARCO": ["train"],  # pairs and mined hard negative.
        "StackExchange": [],  # title body pair.
        "QuoraQA": ["train"],  # duplicate question pairs.
        "MsCocoCaptions": ["train"],  # pairs describe the same image.
        "Flickr30k": ["train"],  # pairs describe the same image.
        "SNLI": ["train"],  # random negative.
        "ESCI": ["train"],  # exact match as positive match and mined hard negative.
        "NegationDataset": [
            "train"
        ],  # synthetically generated negation dataset https://huggingface.co/datasets/jinaai/negation-dataset
        "NQ": ["train"],  # mined hard negative.
        "HotpotQA": ["train"],  # mined hard negative.
        "FEVER": ["train"],  # mined hard negative.
        "CC-NEWS": [],  # title content with random negative.
    },
    public_training_code=None,
    public_training_data=None,
)

jina_embedding_b_en_v1 = ModelMeta(
    loader=partial(
        SentenceTransformerWrapper,
        model_name="jinaai/jina-embedding-b-en-v1",
        revision="32aa658e5ceb90793454d22a57d8e3a14e699516",
    ),
    name="jinaai/jina-embedding-b-en-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="32aa658e5ceb90793454d22a57d8e3a14e699516",
    release_date="2023-07-07",
    n_parameters=110_000_000,
    memory_usage_mb=420,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/jinaai/jina-embedding-b-en-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by="jinaai/jina-embeddings-v2-base-en",
    adapted_from="google-t5/t5-base",
    training_datasets={
        "PAQ": ["train"],
        "GooAQ": ["train"],
        "WikiAnswers": ["train"],
        "AmazonQA": ["train"],
        "ELI5": ["train"],
        "SentenceCompression": ["train"],
        "SimpleWikipedia": ["train"],
        "Specter": ["train"],
        "Squad2": ["train"],
        "Tmdb": ["train"],
        "TrivialQA": ["train"],
        "TweetQA": ["train"],
        "WikiHow": ["train"],
        "Xmarket": [],  # adopted from Cross-Market Recommendation (XMRec).
        "S2ORC": [],  # title abstract pair.
        "YahooAnswers": [],  # question answer pair.
        "MSMARCO": ["train"],  # pairs and mined hard negative.
        "StackExchange": [],  # title body pair.
        "QuoraQA": ["train"],  # duplicate question pairs.
        "MsCocoCaptions": ["train"],  # pairs describe the same image.
        "Flickr30k": ["train"],  # pairs describe the same image.
        "SNLI": ["train"],  # random negative.
        "ESCI": ["train"],  # exact match as positive match and mined hard negative.
        "NegationDataset": [
            "train"
        ],  # synthetically generated negation dataset https://huggingface.co/datasets/jinaai/negation-dataset
    },
    public_training_code=None,
    public_training_data=None,
)

jina_embedding_s_en_v1 = ModelMeta(
    loader=partial(
        SentenceTransformerWrapper,
        model_name="jinaai/jina-embedding-s-en-v1",
        revision="5ac6cd473e2324c6d5f9e558a6a9f65abb57143e",
    ),
    name="jinaai/jina-embedding-s-en-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="5ac6cd473e2324c6d5f9e558a6a9f65abb57143e",
    release_date="2023-07-07",
    n_parameters=35_000_000,
    memory_usage_mb=134,
    embed_dim=512,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/jinaai/jina-embedding-s-en-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by="jinaai/jina-embeddings-v2-small-en",
    adapted_from="google-t5/t5-small",
    training_datasets={
        "PAQ": ["train"],
        "GooAQ": ["train"],
        "WikiAnswers": ["train"],
        "AmazonQA": ["train"],
        "ELI5": ["train"],
        "SentenceCompression": ["train"],
        "SimpleWikipedia": ["train"],
        "Specter": ["train"],
        "Squad2": ["train"],
        "Tmdb": ["train"],
        "TrivialQA": ["train"],
        "TweetQA": ["train"],
        "WikiHow": ["train"],
        "Xmarket": [],  # adopted from Cross-Market Recommendation (XMRec).
        "S2ORC": [],  # title abstract pair.
        "YahooAnswers": [],  # question answer pair.
        "MSMARCO": ["train"],  # pairs and mined hard negative.
        "StackExchange": [],  # title body pair.
        "QuoraQA": ["train"],  # duplicate question pairs.
        "MsCocoCaptions": ["train"],  # pairs describe the same image.
        "Flickr30k": ["train"],  # pairs describe the same image.
        "SNLI": ["train"],  # random negative.
        "ESCI": ["train"],  # exact match as positive match and mined hard negative.
        "NegationDataset": [
            "train"
        ],  # synthetically generated negation dataset https://huggingface.co/datasets/jinaai/negation-dataset
    },
    public_training_code=None,
    public_training_data=None,
)

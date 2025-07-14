from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import torch
from PIL import Image
from sentence_transformers import __version__ as st_version
from transformers import AutoModel

from mteb.encoder_interface import PromptType
from mteb.languages import PROGRAMMING_LANGS
from mteb.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper
from mteb.models.wrapper import Wrapper
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

JinaV4_TRAINING_DATA = {
    "MSMARCO": ["train"],
    "MSMARCOHardNegatives": ["train"],
    "NanoMSMARCORetrieval": ["train"],
    "mMARCO-NL": ["train"],  # translation not trained on
    "NQ": ["train"],
    "NQHardNegatives": ["train"],
    "NanoNQRetrieval": ["train"],
    "NQ-PL": ["train"],  # translation not trained on
    "NQ-NL": ["train"],  # translation not trained on
    "STS12": ["train"],
    "SICK-R": ["train"],
    "CodeSearchNetRetrieval": ["train"],
    "CodeFeedbackST": ["train"],
    "CodeFeedbackMT": ["train"],
    "AppsRetrieval": ["train"],
    "StackOverflowQA": ["train"],
    "CornStack": [],
    "VDRMultilingualRetrieval": ["train"],
    # from https://huggingface.co/datasets/vidore/colpali_train_set
    "DocVQA": ["train"],
    "InfoVQA": ["train"],
    "TATDQA": ["train"],
    "arXivQA": ["train"],
    # "other": [], # inhouse dataset including synthetic datasets
}


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


class JinaV4Wrapper(Wrapper):
    """following the hf model card documentation."""

    SUPPORTED_VECTOR_TYPES = {"single_vector", "multi_vector"}

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code: bool = True,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        requires_package(
            self,
            "flash_attn",
            model,
            "pip install 'mteb[flash_attention]'",
        )
        requires_package(self, "peft", model, "pip install 'mteb[jina-v4]'")
        requires_package(self, "torchvision", model, "pip install 'mteb[jina-v4]'")
        import flash_attn  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401

        self.model = AutoModel.from_pretrained(
            model,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            revision=revision,
        ).eval()
        self.model_prompts = model_prompts or {}
        self.vector_type = "single_vector"  # default vector type

    def _resolve_task_parameters(
        self, task_name: str | None, prompt_type: PromptType | None = None
    ) -> tuple[str, str, str]:
        """Resolve task parameters from task_name and prompt_type.

        Returns:
            tuple: (base_task, prompt_name_param, task_type)
        """
        task_type = self.get_prompt_name(self.model_prompts, task_name, prompt_type)
        jina_task_name = self.model_prompts.get(task_type) if task_type else None

        # Determine prompt name parameter
        if jina_task_name and "query" in jina_task_name:
            prompt_name_param = "query"
        elif jina_task_name and "passage" in jina_task_name:
            prompt_name_param = "passage"
        else:
            prompt_name_param = "query"  # default fallback

        jina_task_name = get_programming_task_override(task_name, jina_task_name)
        # Extract base task (e.g., "retrieval" from "retrieval.query")
        base_task = jina_task_name.split(".")[0] if jina_task_name else "retrieval"

        return base_task, prompt_name_param, task_type

    @staticmethod
    def _log_task_info(
        task_name: str,
        prompt_type: PromptType | None,
        prompt_name: str | None,
        sentences_count: int,
    ) -> None:
        """Log task and prompt information."""
        if prompt_name:
            logger.info(f"Using {prompt_name=} for {task_name=} {prompt_type=}")
        else:
            logger.info(f"No model prompts found for {task_name=} {prompt_type=}")
        logger.info(f"Encoding {sentences_count} sentences.")

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        batch_size: int = 32,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        prompt_name = self.get_prompt_name(self.model_prompts, task_name, prompt_type)
        self._log_task_info(task_name, prompt_type, prompt_name, len(sentences))

        embeddings = self.get_text_embeddings(
            texts=list(sentences),
            task_name=task_name,
            batch_size=batch_size,
            prompt_type=prompt_type,
            return_numpy=True,
            **kwargs,
        )

        return embeddings

    def get_text_embeddings(
        self,
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        return_numpy=False,
        **kwargs: Any,
    ):
        # Resolve task parameters
        base_task, prompt_name_param, task_type = self._resolve_task_parameters(
            task_name, prompt_type
        )

        if task_type.startswith("DocumentUnderstanding"):
            self.vector_type = "multi_vector"
        else:
            self.vector_type = "single_vector"

        with torch.no_grad():
            return self.model.encode_text(
                texts=texts,
                batch_size=batch_size,
                return_multivector=True
                if task_type.startswith("DocumentUnderstanding")
                else False,
                prompt_name=prompt_name_param,
                task=base_task,
                return_numpy=return_numpy,
            )

    def get_image_embeddings(
        self,
        images,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        max_pixels: int = 37788800,
        return_numpy=False,
        **kwargs: Any,
    ):
        import torchvision.transforms.functional as F
        from torch.utils.data import DataLoader

        # Resolve task parameters
        base_task, _, task_type = self._resolve_task_parameters(task_name, prompt_type)

        if task_type.startswith("DocumentUnderstanding"):
            self.vector_type = "multi_vector"
        else:
            self.vector_type = "single_vector"

        all_images = []
        if isinstance(images, DataLoader):
            iterator = images
        else:
            iterator = DataLoader(images, batch_size=batch_size)

        for batch in iterator:
            for b in batch:
                pil_img = (
                    F.to_pil_image(b.to("cpu")) if not isinstance(b, Image.Image) else b
                )
                all_images.append(pil_img)

        batch_size = 1
        return self.model.encode_image(
            images=all_images,
            batch_size=batch_size,
            max_pixels=max_pixels,
            return_multivector=True,
            task=base_task,
            return_numpy=return_numpy,
        )

    def get_fused_embeddings(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(
            "Fused embeddings are not supported yet. Please use get_text_embeddings or get_image_embeddings."
        )

    @staticmethod
    def _convert_to_torch_if_needed(embeddings):
        """Convert numpy arrays to torch tensors if needed."""
        if isinstance(embeddings, np.ndarray):
            return torch.from_numpy(embeddings)
        elif isinstance(embeddings, list):
            # Handle list of numpy arrays or tensors
            converted = []
            for emb in embeddings:
                if isinstance(emb, np.ndarray):
                    converted.append(torch.from_numpy(emb))
                else:
                    converted.append(emb)
            return converted
        return embeddings

    def similarity(self, a, b):
        """Compute similarity between embeddings.

        Args:
            a: First embedding
            b: Second embedding
        """
        a_torch = self._convert_to_torch_if_needed(a)
        b_torch = self._convert_to_torch_if_needed(b)

        if self.vector_type == "single_vector":
            return self.score_single_vector(a_torch, b_torch)
        elif self.vector_type == "multi_vector":
            return self.score_multi_vector(a_torch, b_torch)
        else:
            raise ValueError(
                "vector_type must be one of the following: [`single_vector`, `multi_vector`]"
            )

    @staticmethod
    def score_single_vector(
        qs: torch.Tensor | list[torch.Tensor],
        ps: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the dot product score for the given single-vector query and passage embeddings."""
        device = "cpu"

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        # Normalize inputs to 2D tensors
        def normalize_input(x):
            if isinstance(x, torch.Tensor):
                return x.unsqueeze(0) if x.ndim == 1 else x
            else:  # list
                return torch.stack(x) if len(x) > 1 else x[0].unsqueeze(0)

        qs_stacked = normalize_input(qs).to(device)
        ps_stacked = normalize_input(ps).to(device)

        # Compute scores
        scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked).to(torch.float32)

        # Squeeze if single query
        return scores.squeeze(0) if scores.shape[0] == 1 else scores

    @staticmethod
    def score_multi_vector(
        qs: list[torch.Tensor],
        ps: list[torch.Tensor],
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings."""
        device = "cpu"

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: list[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(
                qs[i : i + batch_size], batch_first=True, padding_value=0
            ).to(device)
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0
                ).to(device)
                scores_batch.append(
                    torch.einsum("bnd,csd->bcns", qs_batch, ps_batch)
                    .max(dim=3)[0]
                    .sum(dim=2)
                )
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), (
            f"Expected {len(qs)} scores, got {scores.shape[0]}"
        )

        scores = scores.to(torch.float32)
        return scores


def get_programming_task_override(
    task_name: str, current_task_name: str | None
) -> str | None:
    """Check if task involves programming content and override with 'code' task if so.

    Args:
        task_name: Original task name to check
        current_task_name: Current Jina task name

    Returns:
        'code' if programming-related task detected, otherwise current_task_name
    """
    # Import here to avoid circular imports
    from mteb import get_task

    task = get_task(task_name)

    # Check various indicators for programming content
    has_code_language = any(lang.endswith("-Code") for lang in task.metadata.eval_langs)
    has_programming_language = any(
        lang in PROGRAMMING_LANGS for lang in task.metadata.languages
    )
    has_programming_domain = any(
        domain == "Programming" for domain in task.metadata.domains
    )

    if has_code_language or has_programming_language or has_programming_domain:
        return "code"

    return current_task_name


jina_embeddings_v4 = ModelMeta(
    loader=partial(  # type: ignore
        JinaV4Wrapper,
        model="jinaai/jina-embeddings-v4",
        revision="26239889730c735ed7e9a4db9180c8935faf4ba0",
        trust_remote_code=True,
        model_prompts={
            "Retrieval-query": "retrieval.query",
            "Retrieval-passage": "retrieval.passage",
            "STS": "text-matching",
            "DocumentUnderstanding": "retrieval.query",
        },
    ),
    name="jinaai/jina-embeddings-v4",
    languages=XLMR_LANGUAGES,
    open_weights=True,
    revision="26239889730c735ed7e9a4db9180c8935faf4ba0",
    release_date="2025-06-24",  # official release date
    modalities=["image", "text"],
    n_parameters=int(3.8 * 1e9),
    memory_usage_mb=7500,
    max_tokens=32768,
    embed_dim=2048,
    license="cc-by-nc-4.0",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    reference="https://huggingface.co/jinaai/jina-embeddings-v4",
    public_training_code=None,
    public_training_data=None,
    training_datasets=JinaV4_TRAINING_DATA,
    adapted_from="Qwen/Qwen2.5-VL-3B-Instruct",
)


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

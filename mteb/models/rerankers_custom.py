from __future__ import annotations

import logging
from functools import partial
from typing import Any, Callable

import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mteb.encoder_interface import Encoder
from mteb.evaluation.evaluators.RetrievalEvaluator import DenseRetrievalExactSearch
from mteb.model_meta import ModelMeta
from mteb.models.bge_models import bge_m3_training_data
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)


class RerankerWrapper(DenseRetrievalExactSearch):
    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 4,
        fp_options: bool = None,
        silent: bool = False,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.fp_options = fp_options if fp_options is not None else torch.float32
        if self.fp_options == "auto":
            self.fp_options = torch.float32
        elif self.fp_options == "float16":
            self.fp_options = torch.float16
        elif self.fp_options == "float32":
            self.fp_options = torch.float32
        elif self.fp_options == "bfloat16":
            self.fp_options = torch.bfloat16
        logger.info(f"Using fp_options of {self.fp_options}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.silent = silent
        self.first_print = True  # for debugging

    def predict(self, input_to_rerank, **kwargs) -> list:
        pass


class BGEReranker(RerankerWrapper):
    name: str = "BGE"

    def __init__(
        self,
        model_name_or_path="BAAI/bge-reranker-v2-m3",
        torch_compile=False,
        **kwargs,
    ):
        super().__init__(model_name_or_path, **kwargs)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {}
        if self.fp_options:
            model_args["torch_dtype"] = self.fp_options

        requires_package(
            self,
            "FlagEmbedding",
            model_name_or_path,
            "pip install 'mteb[flagembedding]'",
        )
        from FlagEmbedding import FlagReranker

        self.model = FlagReranker(model_name_or_path, use_fp16=True)

    @torch.inference_mode()
    def predict(self, input_to_rerank, **kwargs):
        inputs = list(zip(*input_to_rerank))
        if len(input_to_rerank[0]) == 2:
            queries, passages = inputs
            instructions = None
        else:
            queries, passages, instructions = inputs
        if instructions is not None and instructions[0] is not None:
            assert len(instructions) == len(queries)
            queries = [f"{q} {i}".strip() for i, q in zip(instructions, queries)]

        assert len(queries) == len(passages)
        query_passage_tuples = list(zip(queries, passages))
        scores = self.model.compute_score(query_passage_tuples, normalize=True)
        assert len(scores) == len(queries), (
            f"Expected {len(queries)} scores, got {len(scores)}"
        )
        return scores


class MonoBERTReranker(RerankerWrapper):
    name: str = "MonoBERT"

    def __init__(
        self,
        model_name_or_path="castorini/monobert-large-msmarco",
        torch_compile=False,
        **kwargs,
    ):
        super().__init__(model_name_or_path, **kwargs)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {}
        if self.fp_options:
            model_args["torch_dtype"] = self.fp_options
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            **model_args,
        )
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_length = self.tokenizer.model_max_length
        logger.info(f"Using max_length of {self.max_length}")

        self.model.eval()

    @torch.inference_mode()
    def predict(self, input_to_rerank, **kwargs):
        inputs = list(zip(*input_to_rerank))
        if len(input_to_rerank[0]) == 2:
            queries, passages = inputs
            instructions = None
        else:
            queries, passages, instructions = inputs

        if instructions is not None and instructions[0] is not None:
            queries = [f"{q} {i}".strip() for i, q in zip(instructions, queries)]

        tokens = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation="only_second",
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)
        output = self.model(**tokens)[0]
        batch_scores = torch.nn.functional.log_softmax(output, dim=1)
        return batch_scores[:, 1].exp().tolist()


class JinaReranker(RerankerWrapper):
    name = "Jina"

    def __init__(
        self,
        model_name_or_path="jinaai/jina-reranker-v2-base-multilingual",
        torch_compile=False,
        **kwargs,
    ):
        super().__init__(model_name_or_path, **kwargs)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {}
        if self.fp_options:
            model_args["torch_dtype"] = self.fp_options

        self.model = CrossEncoder(
            model_name_or_path,
            automodel_args={"torch_dtype": "auto"},
            trust_remote_code=True,
        )

    def predict(self, input_to_rerank, **kwargs):
        inputs = list(zip(*input_to_rerank))
        if len(input_to_rerank[0]) == 2:
            queries, passages = inputs
            instructions = None
        else:
            queries, passages, instructions = inputs

        if instructions is not None and instructions[0] is not None:
            queries = [f"{q} {i}".strip() for i, q in zip(instructions, queries)]

        if self.first_print:
            logger.info(f"Using {queries[0]}")
            self.first_print = False

        sentence_pairs = list(zip(queries, passages))
        scores = self.model.predict(sentence_pairs, convert_to_tensor=True).tolist()
        return scores


def _loader(wrapper: type[RerankerWrapper], **kwargs) -> Callable[..., Encoder]:
    _kwargs = kwargs

    def loader_inner(**kwargs: Any) -> Encoder:
        return wrapper(**_kwargs, **kwargs)

    return loader_inner()


monobert_large = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=MonoBERTReranker,
        model_name_or_path="castorini/monobert-large-msmarco",
        fp_options="float16",
    ),
    name="castorini/monobert-large-msmarco",
    languages=["eng-Latn"],
    open_weights=True,
    revision="0a97706f3827389da43b83348d5d18c9d53876fa",
    release_date="2020-05-28",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license=None,
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=None,
    training_datasets=None,
    framework=["Sentence Transformers", "PyTorch"],
    is_cross_encoder=True,
)

# languages unclear: https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/discussions/28
jina_reranker_multilingual = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=JinaReranker,
        model_name_or_path="jinaai/jina-reranker-v2-base-multilingual",
        fp_options="float16",
    ),
    name="jinaai/jina-reranker-v2-base-multilingual",
    languages=["eng-Latn"],
    open_weights=True,
    revision="126747772a932960028d9f4dc93bd5d9c4869be4",
    release_date="2024-09-26",
    n_parameters=None,
    memory_usage_mb=531,
    max_tokens=None,
    embed_dim=None,
    license=None,
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=None,
    training_datasets=None,
    framework=["Sentence Transformers", "PyTorch"],
    is_cross_encoder=True,
)

bge_reranker_v2_m3 = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=BGEReranker,
        model_name_or_path="BAAI/bge-reranker-v2-m3",
        fp_options="float16",
    ),
    name="BAAI/bge-reranker-v2-m3",
    languages=[
        "eng-Latn",
        "ara-Arab",
        "ben-Beng",
        "spa-Latn",
        "fas-Arab",
        "fin-Latn",
        "fra-Latn",
        "hin-Deva",
        "ind-Latn",
        "jpn-Jpan",
        "kor-Hang",
        "rus-Cyrl",
        "swa-Latn",
        "tel-Telu",
        "tha-Thai",
        "zho-Hans",
        "deu-Latn",
        "yor-Latn",
        "dan-Latn",
        "heb-Hebr",
        "hun-Latn",
        "ita-Latn",
        "khm-Khmr",
        "msa-Latn",
        "nld-Latn",
        "nob-Latn",
        "pol-Latn",
        "por-Latn",
        "swe-Latn",
        "tur-Latn",
        "vie-Latn",
        "zho-Hant",
    ],
    open_weights=True,
    revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    release_date="2024-06-24",
    n_parameters=None,
    memory_usage_mb=2166,
    max_tokens=None,
    embed_dim=None,
    license=None,
    public_training_code=None,
    public_training_data=None,
    similarity_fn_name=None,
    use_instructions=None,
    training_datasets=bge_m3_training_data,
    framework=["Sentence Transformers", "PyTorch"],
    is_cross_encoder=True,
)

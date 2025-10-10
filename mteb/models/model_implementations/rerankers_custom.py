import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

from mteb._requires_package import requires_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta
from mteb.types import Array, BatchedInput, PromptType

from .bge_models import bge_m3_training_data

logger = logging.getLogger(__name__)


class RerankerWrapper:
    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 4,
        fp_options: bool | None = None,
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
    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        queries = [text for batch in inputs1 for text in batch["query"]]
        instructions = None
        if "instruction" in inputs2.dataset.features:
            instructions = [text for batch in inputs1 for text in batch["instruction"]]
        passages = [text for batch in inputs2 for text in batch["text"]]

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
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        queries = [text for batch in inputs1 for text in batch["query"]]
        instructions = None
        if "instruction" in inputs2.dataset.features:
            instructions = [text for batch in inputs1 for text in batch["instruction"]]
        passages = [text for batch in inputs2 for text in batch["text"]]

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
        return batch_scores[:, 1].exp()


class JinaReranker(RerankerWrapper):
    name = "Jina"

    def __init__(
        self,
        model_name_or_path="jinaai/jina-reranker-v2-base-multilingual",
        torch_compile=False,
        **kwargs,
    ):
        from sentence_transformers import CrossEncoder

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

    @torch.inference_mode()
    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        queries = [text for batch in inputs1 for text in batch["query"]]
        instructions = None
        if "instruction" in inputs2.dataset.features:
            instructions = [text for batch in inputs1 for text in batch["instruction"]]
        passages = [text for batch in inputs2 for text in batch["text"]]

        if instructions is not None and instructions[0] is not None:
            queries = [f"{q} {i}".strip() for i, q in zip(instructions, queries)]

        if self.first_print:
            logger.info(f"Using {queries[0]}")
            self.first_print = False

        sentence_pairs = list(zip(queries, passages))
        scores = self.model.predict(sentence_pairs, convert_to_tensor=True)
        return scores


monobert_large = ModelMeta(
    loader=MonoBERTReranker,  # type: ignore
    loader_kwargs=dict(
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
    loader=JinaReranker,  # type: ignore
    loader_kwargs=dict(
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
    loader=BGEReranker,  # type: ignore
    loader_kwargs=dict(
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
    citation="""
    @misc{li2023making,
      title={Making Large Language Models A Better Foundation For Dense Retrieval},
      author={Chaofan Li and Zheng Liu and Shitao Xiao and Yingxia Shao},
      year={2023},
      eprint={2312.15503},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }
    @misc{chen2024bge,
          title={BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
          author={Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu},
          year={2024},
          eprint={2402.03216},
          archivePrefix={arXiv},
          primaryClass={cs.CL}
    }
    """,
)

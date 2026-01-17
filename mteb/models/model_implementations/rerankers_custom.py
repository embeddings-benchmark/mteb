import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

from mteb._requires_package import requires_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta, ScoringFunction
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


class Qwen3RerankerWrapper:
    """Wrapper for Qwen3 Reranker models."""

    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype=torch.float32,
        attn_implementation: str | None = None,
        batch_size: int = 32,
        max_length: int = 8192,
        **kwargs,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_kwargs = {"torch_dtype": torch_dtype}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        logger.info(f"Loading Qwen3 Reranker model: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **model_kwargs
        )
        self.model.to(self.device)
        self.model.eval()

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(
            self.prefix, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            self.suffix, add_special_tokens=False
        )

    @staticmethod
    def format_instruction(instruction: str | None, query: str, doc: str) -> str:
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        output = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        return output

    def process_inputs(self, pairs: list[str]) -> dict:
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length
            - len(self.prefix_tokens)
            - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens

        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=self.max_length
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs: dict) -> list[float]:
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

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

        all_scores = []
        for i in range(0, len(queries), self.batch_size):
            batch_queries = queries[i : i + self.batch_size]
            batch_passages = passages[i : i + self.batch_size]
            batch_instructions = (
                instructions[i : i + self.batch_size]
                if instructions is not None
                else [None] * len(batch_queries)
            )

            pairs = [
                self.format_instruction(instr, query, doc)
                for instr, query, doc in zip(
                    batch_instructions, batch_queries, batch_passages
                )
            ]

            inputs = self.process_inputs(pairs)
            scores = self.compute_logits(inputs)
            all_scores.extend(scores)

        return all_scores


monobert_large = ModelMeta(
    loader=MonoBERTReranker,
    loader_kwargs=dict(
        fp_options="float16",
    ),
    name="castorini/monobert-large-msmarco",
    model_type=["cross-encoder"],
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
    framework=["Sentence Transformers", "PyTorch", "Transformers"],
)

# languages unclear: https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/discussions/28
jina_reranker_multilingual = ModelMeta(
    loader=JinaReranker,
    loader_kwargs=dict(
        fp_options="float16",
    ),
    name="jinaai/jina-reranker-v2-base-multilingual",
    model_type=["cross-encoder"],
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
    framework=[
        "Sentence Transformers",
        "PyTorch",
        "Transformers",
        "ONNX",
        "safetensors",
    ],
)

bge_reranker_v2_m3 = ModelMeta(
    loader=BGEReranker,
    loader_kwargs=dict(
        fp_options="float16",
    ),
    name="BAAI/bge-reranker-v2-m3",
    model_type=["cross-encoder"],
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
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
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

qwen3_reranker_training_data = {
    # source: https://arxiv.org/pdf/2506.05176
    "MIRACLReranking",
    "DuRetrieval",
    "MrTidyRetrieval",
    "T2Reranking",
    "MSMARCO",
    "NQ",
    "HotpotQA",
    "CodeSearchNet",
    "MultiLongDocRetrieval",
    # "NLI",
    # "simclue",
    # "multi-cpr",
    # + synthetic data
}

qwen3_reranker_0_6b = ModelMeta(
    loader=Qwen3RerankerWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    ),
    name="Qwen/Qwen3-Reranker-0.6B",
    revision="6e9e69830b95c52b5fd889b7690dda3329508de3",
    release_date="2025-05-29",
    languages=None,
    n_parameters=595776512,
    memory_usage_mb=1136.0,
    max_tokens=40960,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Qwen/Qwen3-Reranker-0.6B",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=qwen3_reranker_training_data,
    adapted_from=None,
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation="""@article{qwen3embedding,
  title={Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models},
  author={Zhang, Yanzhao and Li, Mingxin and Long, Dingkun and Zhang, Xin and Lin, Huan and Yang, Baosong and Xie, Pengjun and Yang, An and Liu, Dayiheng and Lin, Junyang and Huang, Fei and Zhou, Jingren},
  journal={arXiv preprint arXiv:2506.05176},
  year={2025}
}""",
    contacts=None,
)

qwen3_reranker_4b = ModelMeta(
    loader=Qwen3RerankerWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    ),
    name="Qwen/Qwen3-Reranker-4B",
    revision="f16fc5d5d2b9b1d0db8280929242745d79794ef5",
    release_date="2025-06-03",
    languages=None,
    n_parameters=4021784576,
    memory_usage_mb=7671.0,
    max_tokens=40960,
    embed_dim=2560,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Qwen/Qwen3-Reranker-4B",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=qwen3_reranker_training_data,
    adapted_from=None,
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation="""@article{qwen3embedding,
  title={Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models},
  author={Zhang, Yanzhao and Li, Mingxin and Long, Dingkun and Zhang, Xin and Lin, Huan and Yang, Baosong and Xie, Pengjun and Yang, An and Liu, Dayiheng and Lin, Junyang and Huang, Fei and Zhou, Jingren},
  journal={arXiv preprint arXiv:2506.05176},
  year={2025}
}""",
    contacts=None,
)

qwen3_reranker_8b = ModelMeta(
    loader=Qwen3RerankerWrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    ),
    name="Qwen/Qwen3-Reranker-8B",
    revision="5fa94080caafeaa45a15d11f969d7978e087a3db",
    release_date="2025-05-29",
    languages=None,
    n_parameters=8188548096,
    memory_usage_mb=15618.0,
    max_tokens=40960,
    embed_dim=4096,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Qwen/Qwen3-Reranker-8B",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=qwen3_reranker_training_data,
    adapted_from=None,
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation="""@article{qwen3embedding,
  title={Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models},
  author={Zhang, Yanzhao and Li, Mingxin and Long, Dingkun and Zhang, Xin and Lin, Huan and Yang, Baosong and Xie, Pengjun and Yang, An and Liu, Dayiheng and Lin, Junyang and Huang, Fei and Zhou, Jingren},
  journal={arXiv preprint arXiv:2506.05176},
  year={2025}
}""",
    contacts=None,
)

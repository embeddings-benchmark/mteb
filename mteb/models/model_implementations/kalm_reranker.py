from __future__ import annotations
from typing import TYPE_CHECKING
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from mteb.models.model_meta import ModelMeta

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput


if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

DEFAULT_INSTRUCTION = "Given a query, retrieve documents that answer the query."
DEFAULT_SYSTEM_INSTRUCTION = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)


class KaLMRerankerWrapper:
    """Wrapper for KaLM-Reranker-V1 models.

    Reference implementation https://huggingface.co/KaLM-Embedding/KaLM-Reranker-V1-Nano/blob/main/kalm_reranker.py

    Score query-document relevance with a KaLM encoder-decoder reranker.
    The returned score is ``P(yes)`` after applying a two-class softmax to the
    model's ``yes`` and ``no`` logits.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        batch_size: int = 32,
        query_max_length: int = 512,
        max_length: int = 1024,
        chunk_size: Optional[int] = 4,
        instruction: str = DEFAULT_INSTRUCTION,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        **model_kwargs: Any,
    ) -> None:
        if not isinstance(model_name_or_path, str) or not model_name_or_path:
            raise ValueError("model_name_or_path must be a non-empty string.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if query_max_length <= 0 or max_length <= 0:
            raise ValueError("query_max_length and max_length must be positive.")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive or None.")
        if not isinstance(instruction, str) or not isinstance(system_instruction, str):
            raise TypeError("instruction and system_instruction must be strings.")

        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype, self.device)
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.instruction = instruction
        self.system_instruction = system_instruction

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(
                    "The tokenizer must define a pad token or an EOS token."
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            dtype=self.dtype,
            **model_kwargs,
        )

        for parameter in self.model.parameters():
            if parameter.is_floating_point() and parameter.dtype != self.dtype:
                parameter.data = parameter.data.to(dtype=self.dtype)
        self.model.to(device=self.device)
        self.model.eval()

        self.yes_token_id = self._answer_token_id("yes")
        self.no_token_id = self._answer_token_id("no")

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return resolved

    @staticmethod
    def _resolve_dtype(
        dtype: Optional[Union[str, torch.dtype]], device: torch.device
    ) -> torch.dtype:
        if dtype is None:
            return torch.bfloat16 if device.type == "cuda" else torch.float32
        if isinstance(dtype, torch.dtype):
            return dtype
        if not isinstance(dtype, str):
            raise TypeError(
                "dtype must be a torch.dtype or a string such as 'bfloat16'."
            )
        normalized = dtype.lower().removeprefix("torch.")
        supported = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized not in supported:
            raise ValueError(f"Unsupported dtype: {dtype!r}.")
        return supported[normalized]

    def _answer_token_id(self, answer: str) -> int:
        token_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError(f"Failed to tokenize the answer {answer!r}.")
        return token_ids[-1]

    def _get_encoder(self):
        if hasattr(self.model, "get_encoder"):
            return self.model.get_encoder()
        if hasattr(self.model, "encoder"):
            return self.model.encoder
        raise AttributeError(f"Cannot find the encoder on {type(self.model).__name__}.")

    @staticmethod
    def _pool_encoder_chunks(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        num_chunks = (sequence_length + chunk_size - 1) // chunk_size
        padded_length = num_chunks * chunk_size
        pad_length = padded_length - sequence_length

        if pad_length:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length))
            attention_mask = F.pad(attention_mask, (0, pad_length))

        hidden_states = hidden_states.view(
            batch_size, num_chunks, chunk_size, hidden_size
        )
        chunk_mask = attention_mask.view(batch_size, num_chunks, chunk_size)
        expanded_mask = chunk_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled_hidden = (hidden_states * expanded_mask).sum(dim=2)
        pooled_hidden = pooled_hidden / chunk_mask.sum(dim=2).clamp(min=1).unsqueeze(-1)
        pooled_mask = (chunk_mask.sum(dim=2) > 0).to(attention_mask.dtype)
        return pooled_hidden, pooled_mask

    def _decoder_text(self, query: str, instruction: str) -> str:
        query_ids = self.tokenizer(
            query,
            add_special_tokens=False,
            truncation=True,
            max_length=self.query_max_length,
        )["input_ids"]
        truncated_query = self.tokenizer.decode(
            query_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return (
            "<bos><start_of_turn>user\n"
            f"{self.system_instruction}\n\n"
            f"<Instruct>: {instruction}\n"
            f"<Query>: {truncated_query}<end_of_turn>\n"
            "<start_of_turn>model\n\n\n\n"
        )

    @staticmethod
    def _validate_pairs(
        pairs: Sequence[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
            raise TypeError("pairs must be a sequence of (query, document) pairs.")
        validated: List[Tuple[str, str]] = []
        for index, pair in enumerate(pairs):
            if (
                isinstance(pair, (str, bytes))
                or not isinstance(pair, Sequence)
                or len(pair) != 2
            ):
                raise ValueError(f"pairs[{index}] must contain exactly two strings.")
            query, document = pair
            if not isinstance(query, str) or not isinstance(document, str):
                raise TypeError(f"pairs[{index}] must contain exactly two strings.")
            validated.append((query, document))
        return validated

    @torch.inference_mode()
    def _predict_batch(
        self, pairs: Sequence[Tuple[str, str]], instruction: str
    ) -> List[float]:
        encoder_texts = [f"<Document>: {document}" for _, document in pairs]
        decoder_texts = [self._decoder_text(query, instruction) for query, _ in pairs]

        encoder_batch = self.tokenizer(
            encoder_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        decoder_batch = self.tokenizer(
            decoder_texts,
            padding=True,
            pad_to_multiple_of=8,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        if self.chunk_size is None:
            outputs = self.model(
                input_ids=encoder_batch["input_ids"],
                attention_mask=encoder_batch["attention_mask"],
                decoder_input_ids=decoder_batch["input_ids"],
                decoder_attention_mask=decoder_batch["attention_mask"],
                return_dict=True,
            )
        else:
            encoder_outputs = self._get_encoder()(
                input_ids=encoder_batch["input_ids"],
                attention_mask=encoder_batch["attention_mask"],
                return_dict=True,
            )
            pooled_hidden, pooled_mask = self._pool_encoder_chunks(
                encoder_outputs.last_hidden_state,
                encoder_batch["attention_mask"],
                self.chunk_size,
            )
            outputs = self.model(
                encoder_outputs=BaseModelOutput(last_hidden_state=pooled_hidden),
                attention_mask=pooled_mask,
                decoder_input_ids=decoder_batch["input_ids"],
                decoder_attention_mask=decoder_batch["attention_mask"],
                return_dict=True,
            )

        sequence_lengths = decoder_batch["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(outputs.logits.shape[0], device=self.device)
        last_logits = outputs.logits[batch_indices, sequence_lengths]
        yes_no_logits = torch.stack(
            (
                last_logits[:, self.yes_token_id],
                last_logits[:, self.no_token_id],
            ),
            dim=-1,
        ).float()
        if not torch.isfinite(yes_no_logits).all():
            bad_count = (~torch.isfinite(yes_no_logits).all(dim=-1)).sum().item()
            raise RuntimeError(
                f"The model produced non-finite yes/no logits for {bad_count} input(s). "
                "Use bfloat16 or float32 instead of float16."
            )
        return torch.softmax(yes_no_logits, dim=-1)[:, 0].cpu().tolist()

    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        instruction: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> Array:
        """Return ``P(yes)`` scores in the same order as ``pairs``."""
        queries = [text for batch in inputs1 for text in batch["text"]]
        documents = [text for batch in inputs2 for text in batch["text"]]
        pairs = [(query, document) for query, document in zip(queries, documents)]
        validated_pairs = self._validate_pairs(pairs)
        if not validated_pairs:
            return []
        effective_instruction = self.instruction if instruction is None else instruction
        if not isinstance(effective_instruction, str):
            raise TypeError("instruction must be a string or None.")
        effective_batch_size = self.batch_size if batch_size is None else batch_size
        if not isinstance(effective_batch_size, int) or effective_batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        length_sorted_indices = np.argsort(
            [-(len(query) + len(document)) for query, document in validated_pairs]
        )
        sorted_pairs = [validated_pairs[index] for index in length_sorted_indices]

        tested_batch_size = effective_batch_size
        while tested_batch_size > 1:
            try:
                self._predict_batch(
                    sorted_pairs[: min(len(sorted_pairs), tested_batch_size)],
                    effective_instruction,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                tested_batch_size = max(1, tested_batch_size * 3 // 4)

        sorted_scores: List[float] = []
        try:
            for start in range(0, len(sorted_pairs), tested_batch_size):
                sorted_scores.extend(
                    self._predict_batch(
                        sorted_pairs[start : start + tested_batch_size],
                        effective_instruction,
                    )
                )
        except torch.cuda.OutOfMemoryError as error:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA ran out of memory during reranking. Retry with a smaller batch_size "
                "or shorter max_length."
            ) from error
        inverse_indices = np.argsort(length_sorted_indices)
        return np.array([sorted_scores[index] for index in inverse_indices])


kalm_reranker_v1_training_data = {
    "AdvertiseGen",
    "CHEF",
    "CodeFeedback",
    "DRCD",
    "Expertqa",
    "GooAQ",
    "LCSTS",
    "MEDI2BGE",
    "Multi-CPR",
    "OpenOrca",
    "PAQ",
    "PubMedQA",
    "RefGPT",
    "SearchQA",
    "T2Ranking",
    "THUCNews",
    "UMETRIP-QA",
    "WebCPM",
    "arxiv_qa",
    "aya_dataset",
    "cCOVID-News",
    "cMedQA-V2.0",
    "ccnews",
    "CMRC 2018",
    "cord19_trec-covid",
    "CQADupstack",
    "csl",
    "dbpedia-entity",
    "dureader",
    "dureader-checklist",
    "esci",
    "fever",
    "fiqa",
    "hotpot_qa",
    "law-gpt",
    "lawzhidao",
    "lima-chinese",
    "miracl",
    "mmarco-chinese",
    "mr-tydi",
    "msmarco-passage",
    "msmarco-v2",
    "nfcorpus",
    "rag-dataset-12000",
    "retrieval_data_llm_infgrad",
    "scifact",
    "squad_v2",
    "triviaqa",
    "webgpt_comparisons",
    "webqa",
    "wikipedia-nq",
    "yahoo-answers",
    "quora-question-pairs",
    "arguana",
}

multilingual_langs = [
    "afr-Latn",
    "ara-Arab",
    "aze-Latn",
    "bel-Cyrl",
    "bul-Cyrl",
    "ben-Beng",
    "cat-Latn",
    "ceb-Latn",
    "ces-Latn",
    "cym-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "spa-Latn",
    "est-Latn",
    "eus-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "glg-Latn",
    "guj-Gujr",
    "heb-Hebr",
    "hin-Deva",
    "hrv-Latn",
    "hat-Latn",
    "hun-Latn",
    "hye-Armn",
    "ind-Latn",
    "isl-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "jav-Latn",
    "kat-Geor",
    "kaz-Cyrl",
    "khm-Khmr",
    "kan-Knda",
    "kor-Hang",
    "kir-Cyrl",
    "lao-Laoo",
    "lit-Latn",
    "lav-Latn",
    "mkd-Cyrl",
    "mal-Mlym",
    "mon-Cyrl",
    "mar-Deva",
    "msa-Latn",
    "mya-Mymr",
    "nep-Deva",
    "nld-Latn",
    "nor-Latn",
    "nob-Latn",
    "nno-Latn",
    "pan-Guru",
    "pol-Latn",
    "por-Latn",
    "que-Latn",
    "ron-Latn",
    "rus-Cyrl",
    "sin-Sinh",
    "slk-Latn",
    "slv-Latn",
    "swa-Latn",
    "tam-Taml",
    "tel-Telu",
    "tha-Thai",
    "tgl-Latn",
    "tur-Latn",
    "ukr-Cyrl",
    "urd-Arab",
    "vie-Latn",
    "yor-Latn",
    "zho-Hans",
]

KALM_RERANKER_V1_CITATION = """@misc{zhao2026kalmrerankerv1,
      title={KaLM-Reranker-V1: Fast but Not Late Interaction for Compressed Document Reranking}, 
      author={Xinping Zhao and Jiaxin Xu and Ziqi Dai and Xin Zhang and Shouzheng Huang and Danyu Tang and Xinshuo Hu and Meishan Zhang and Baotian Hu and Min Zhang},
      year={2026},
      eprint={2606.22807},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.22807}, 
}"""


kalm_reranker_v1_nano = ModelMeta(
    loader=KaLMRerankerWrapper,
    loader_kwargs=dict(
        dtype=torch.bfloat16,
    ),
    name="KaLM-Embedding/KaLM-Reranker-V1-Nano",
    revision="1d3dcd79115a77b91b2ece798f536880d2115e48",
    release_date="2026-06-23",
    languages=multilingual_langs,
    n_parameters=786029296,
    n_embedding_parameters=167772160,
    memory_usage_mb=2998.0,
    max_tokens=131072,
    embed_dim=640,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/KaLM-Embedding/KaLM-Reranker-V1-Nano",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=kalm_reranker_v1_training_data,
    adapted_from="google/t5gemma-2-270m-270m",
    superseded_by=None,
    modalities=["text"],
    model_type=["cross-encoder"],
    citation=KALM_RERANKER_V1_CITATION,
    contacts=None,
)


kalm_reranker_v1_small = ModelMeta(
    loader=KaLMRerankerWrapper,
    loader_kwargs=dict(
        dtype=torch.bfloat16,
    ),
    name="KaLM-Embedding/KaLM-Reranker-V1-Small",
    revision="e8eaadc957a7ae383a4983a483c2c399f1056cd3",
    release_date="2026-06-23",
    languages=multilingual_langs,
    n_parameters=2115977456,
    n_embedding_parameters=301989888,
    memory_usage_mb=8072.0,
    max_tokens=131072,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/KaLM-Embedding/KaLM-Reranker-V1-Small",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=kalm_reranker_v1_training_data,
    adapted_from="google/t5gemma-2-1b-1b",
    superseded_by=None,
    modalities=["text"],
    model_type=["cross-encoder"],
    citation=KALM_RERANKER_V1_CITATION,
    contacts=None,
)

kalm_reranker_v1_large = ModelMeta(
    loader=KaLMRerankerWrapper,
    loader_kwargs=dict(
        dtype=torch.bfloat16,
    ),
    name="KaLM-Embedding/KaLM-Reranker-V1-Large",
    revision="bc5cb8fe5a266b6d0b5ffdb9da4c06c950ae242f",
    release_date="2026-06-23",
    languages=multilingual_langs,
    n_parameters=7508928880,
    n_embedding_parameters=671088640,
    memory_usage_mb=28644.0,
    max_tokens=131072,
    embed_dim=2560,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/KaLM-Embedding/KaLM-Reranker-V1-Large",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets=kalm_reranker_v1_training_data,
    adapted_from="google/t5gemma-2-4b-4b",
    superseded_by=None,
    modalities=["text"],
    model_type=["cross-encoder"],
    citation=KALM_RERANKER_V1_CITATION,
    contacts=None,
)

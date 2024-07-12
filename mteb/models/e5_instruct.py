from __future__ import annotations

import logging
from itertools import islice
from typing import Any, Callable, Iterable, Literal, Optional, Sequence, Type, TypeVar

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BatchEncoding
from transformers.modeling_outputs import ModelOutput

from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta

from .e5_models import E5_PAPER_RELEASE_DATE, XLMR_LANGUAGES
from .instructions import task_to_instruction

logger = logging.getLogger(__name__)

T = TypeVar("T")
EncodeTypes = Literal["query", "passage"]

MISTRAL_LANGUAGES = ["eng_Latn", "fra_Latn", "deu_Latn", "ita_Latn", "spa_Latn"]


def batched(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    """batched('ABCDEFG', 3) --> ABC DEF G"""  # noqa
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class E5InstructWrapper(Encoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        max_length: int,
        max_batch_size: Optional[int] = None,
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs: Any,
    ):
        logger.info("Started loading e5 instruct model")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, **kwargs
        )

        self.model = AutoModel.from_pretrained(model_name, **kwargs).to(device)
        self.max_length = max_length
        self.max_batch_size = max_batch_size

    def preprocess(
        self, sentences: Sequence[str], instruction: str, encode_type: EncodeTypes
    ) -> BatchEncoding:
        if encode_type == "query":
            sentences = [
                f"Instruction: {instruction}\nQuery: {sentence}"
                for sentence in sentences
            ]

        batch_dict = self.tokenizer(
            sentences,  # type: ignore
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return batch_dict.to(self.model.device)

    def get_embedding_from_output(
        self, output: ModelOutput, batch_dict: BatchEncoding
    ) -> torch.Tensor:
        return self.average_pool(output.last_hidden_state, batch_dict["attention_mask"])  # type: ignore

    @staticmethod
    def average_pool(
        last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(
        self,
        sentences: list[str],
        *,
        prompt_name: str | None = None,
        batch_size: int = 32,
        encode_type: EncodeTypes = "query",
        **kwargs: Any,  # noqa
    ) -> np.ndarray:
        if self.max_batch_size and batch_size > self.max_batch_size:
            batch_size = self.max_batch_size
        batched_embeddings = []
        if prompt_name is not None:
            instruction = task_to_instruction(
                prompt_name, is_query=encode_type == "query"
            )
        else:
            instruction = ""
        for batch in tqdm(batched(sentences, batch_size)):
            with torch.inference_mode():
                batch_dict = self.preprocess(
                    batch, instruction=instruction, encode_type=encode_type
                )
                outputs = self.model(**batch_dict)
                embeddings = self.get_embedding_from_output(outputs, batch_dict)
            batched_embeddings.append(embeddings.detach().cpu())

        return torch.cat(batched_embeddings).to("cpu").detach().numpy()

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list[str]] | list[str],
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        sep = " "
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()  # type: ignore
                for i in range(len(corpus["text"]))  # type: ignore
            ]
        else:
            if isinstance(corpus[0], str):
                sentences = corpus
            else:
                sentences = [
                    (doc["title"] + sep + doc["text"]).strip()
                    if "title" in doc
                    else doc["text"].strip()
                    for doc in corpus
                ]
        return self.encode(
            sentences, encode_type="passage", prompt_name=prompt_name, **kwargs
        )

    def encode_queries(
        self, queries: list[str], prompt_name: str | None = None, **kwargs: Any
    ) -> np.ndarray:
        return self.encode(
            queries, encode_type="query", prompt_name=prompt_name, **kwargs
        )


class E5MistralWrapper(E5InstructWrapper):
    def __init__(
        self,
        name: str,
        revision: str,
        max_batch_size: int = 4,
        torch_dtype=torch.float16,
        **kwargs,
    ):
        assert (
            name == "intfloat/e5-mistral-7b-instruct"
        ), f"Unexpected model name: {name}"
        super().__init__(
            model_name=name,
            revision=revision,
            max_length=4096,
            max_batch_size=max_batch_size,
            torch_dtype=torch_dtype,
            **kwargs,
        )

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def get_embbeding_from_output(
        self, output: ModelOutput, batch_dict: BatchEncoding
    ) -> torch.Tensor:
        return self.last_token_pool(
            output.last_hidden_state,  # type: ignore
            batch_dict["attention_mask"],  # type: ignore
        )

    def preprocess(
        self, sentences: Sequence[str], instruction: str, encode_type: EncodeTypes
    ) -> BatchEncoding:
        if encode_type == "query":
            sentences = [
                f"Instruction: {instruction}\nQuery: {sentence}"
                for sentence in sentences
            ]
        batch_dict: BatchEncoding = self.tokenizer(
            sentences,  # type: ignore
            max_length=self.max_length - 1,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        # append eos_token_id to every input_ids
        batch_dict["input_ids"] = [
            [*input_ids, self.tokenizer.eos_token_id]
            for input_ids in batch_dict["input_ids"]  # type: ignore
        ]
        batch_dict = self.tokenizer.pad(
            batch_dict, padding=True, return_attention_mask=True, return_tensors="pt"
        )

        return batch_dict.to(self.model.device)


def _loader(
    wrapper: Type[E5InstructWrapper], name: str, revision: str, **kwargs
) -> Callable[..., Encoder]:
    _kwargs = kwargs

    def loader_inner(**kwargs: Any) -> Encoder:
        return wrapper(name, revision=revision, **_kwargs, **kwargs)

    return loader_inner


e5_instruct = ModelMeta(
    loader=_loader(
        E5InstructWrapper,
        "intfloat/multilingual-e5-large-instruct",
        "baa7be480a7de1539afce709c8f13f833a510e0a",
        max_length=512,
    ),
    name="intfloat/multilingual-e5-large-instruct",
    languages=XLMR_LANGUAGES,
    open_source=True,
    revision="baa7be480a7de1539afce709c8f13f833a510e0a",
    release_date=E5_PAPER_RELEASE_DATE,
)

e5_mistral = ModelMeta(
    loader=_loader(
        E5MistralWrapper,
        "intfloat/e5-mistral-7b-instruct",
        "07163b72af1488142a360786df853f237b1a3ca1",
        max_batch_size=4,
        torch_dtype=torch.float16,
    ),
    name="intfloat/e5-mistral-7b-instruct",
    languages=XLMR_LANGUAGES,
    open_source=True,
    revision="07163b72af1488142a360786df853f237b1a3ca1",
    release_date=E5_PAPER_RELEASE_DATE,
)

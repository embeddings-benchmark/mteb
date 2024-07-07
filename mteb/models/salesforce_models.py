from __future__ import annotations

from functools import partial
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts

try:
    from .instructions import task_to_instruction
except:
    from instructions import task_to_instruction

class SFRWrapperV0:
    """Follow the implementation from https://huggingface.co/Salesforce/SFR-Embedding-2_R"""

    def __init__(self, model_name: str, **kwargs: Any):
        self.model_name = model_name
        self.mdl = SentenceTransformer(model_name)

    def to(self, device: torch.device) -> None:
        self.mdl.to(device)

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        is_query = kwargs.pop("is_query", True)
        if "prompt_name" in kwargs:
            instruction = task_to_instruction(kwargs.pop("prompt_name"), is_query)
            sentences = [self.get_detailed_instruct(instruction, q) for q in sentences]
        return self.mdl.encode(sentences, batch_size=batch_size, **kwargs)

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"

    def encode_queries(self, queries: list[str], batch_size: int = 32, **kwargs: Any):
        instruction = ""
        if "instruction" in kwargs:
            instruction = kwargs.pop("instruction")

        sentences = [self.get_detailed_instruct(instruction, q) for q in queries]
        emb = self.encode(sentences, batch_size=batch_size, **kwargs)
        return emb

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list[str]],
        batch_size: int = 32,
        **kwargs: Any,
    ):
        kwargs["is_query"] = False
        sentences = corpus_to_texts(corpus)
        emb = self.encode(sentences, batch_size=batch_size, **kwargs)
        return emb

class SFRWrapperV1:
    """Based on https://huggingface.co/Salesforce/SFR-Embedding-2_R"""
    def __init__(self, model_name: str, **kwargs: Any):
        self.model_name = model_name
        self.mdl = AutoModel.from_pretrained(model_name)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            print(f"----------Using {self.gpu_count} data-parallel GPUs----------")
            self.model = torch.nn.DataParallel(self.model)

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
            max_length=4096,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return batch_dict.to(self.device)

    def get_embedding_from_output(
        self, output: ModelOutput, batch_dict: BatchEncoding
    ) -> torch.Tensor:
        return  F.normalize(last_token_pool(output.last_hidden_state, batch_dict["attention_mask"]), p=2, dim=1)  # type: ignore

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        if "prompt_name" in kwargs:
            instruction = task_to_instruction(
                kwargs.pop("prompt_name"), kwargs.get("is_query", True)
            )
            sentences = [self.get_detailed_instruct(instruction, q) for q in sentences]
        kwargs.pop("is_query", None)
        return self.mdl.encode(sentences, batch_size=batch_size, **kwargs)

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"

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
        batch_size = batch_size * self.gpu_count
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

def sfr_loader(**kwargs):
    try:
        from gritlm import GritLM
        class SFRWrapper(GritLM):
            def get_detailed_instruct(self, instruction: str, query: str) -> str:
                return f"Instruct: {instruction}\nQuery: "

            def encode(self, *args, **kwargs):
                instruction = ""
                if ("prompt_name" in kwargs) and (kwargs.get("is_query", True)):
                    instruction = self.get_detailed_instruct(
                        task_to_instruction(kwargs.pop("prompt_name"))
                    )
                kwargs["instruction"] = instruction
                return super().encode(*args, **kwargs)

            def encode_corpus(self, *args, **kwargs):
                kwargs["is_query"] = False
                return super().encode_corpus(*args, **kwargs)
    except ImportError:
        raise ImportError(
            "Please install `pip install gritlm` to use SFR_Embedding_2_R."
        )
    kwargs.pop("device", None)  # GritLM does automatic device placement
    return SFRWrapper(**kwargs)

SFR_Embedding_2_R = ModelMeta(
    loader=partial(
        sfr_loader,
        model_name_or_path="Salesforce/SFR-Embedding-2_R",
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype=torch.bfloat16,
        # The ST script does not normalize while the HF one does so unclear what to do
        # https://huggingface.co/Salesforce/SFR-Embedding-2_R
        normalized=True,
    ),
    name="Salesforce/SFR-Embedding-2_R",
    languages=["eng_Latn"],
    open_source=True,
    revision="33888956c27c1f0a14edc7f8412c54ca54bb54c3",
    release_date="2024-06-14",  # initial commit of hf model.
)
"""
SFR_Embedding_2_R = ModelMeta(
    loader=partial(SFRWrapperV0, model_name="Salesforce/SFR-Embedding-2_R"),
    name="Salesforce/SFR-Embedding-2_R",
    languages=["eng_Latn"],
    open_source=True,
    revision="33888956c27c1f0a14edc7f8412c54ca54bb54c3",
    release_date="2024-06-14",  # initial commit of hf model.
)
"""

if __name__ == "__main__":
    import mteb

    mdl = mteb.get_model(SFR_Embedding_2_R.name, SFR_Embedding_2_R.revision)
    emb = mdl.encode(["Hello, world!"])
    print(emb)

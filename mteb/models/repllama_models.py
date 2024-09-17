from __future__ import annotations

import logging
from typing import Any, Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from transformers import AutoModel, AutoTokenizer

from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EncodeTypes = Literal["query", "passage"]


class RepLLaMAWrapper:
    def __init__(self, *args, **kwargs):
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError(
                "To use the RepLLaMA based models `peft` is required. Please install it with `pip install 'mteb[peft]'`."
            )

        self.base_model = AutoModel.from_pretrained(
            kwargs["base_model_name_or_path"],
            torch_dtype=kwargs["torch_dtype"],
            device_map=kwargs["device_map"],
        )
        self.model = PeftModel.from_pretrained(
            self.base_model, kwargs["peft_model_name_or_path"]
        )
        self.model = self.model.merge_and_unload()

        self.tokenizer = AutoTokenizer.from_pretrained(
            kwargs["base_model_name_or_path"]
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        # set the max_length for the evals as they did, although the model can handle longer
        self.model.config.max_length = 512
        self.tokenizer.model_max_length = 512

    def create_batch_dict(self, tokenizer, input_texts):
        max_length = self.model.config.max_length
        batch_dict = tokenizer(
            input_texts,
            max_length=max_length - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        batch_dict["input_ids"] = [
            input_ids + [tokenizer.eos_token_id]
            for input_ids in batch_dict["input_ids"]
        ]
        return tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def encode(
        self,
        sentences: list[str],
        *,
        prompt_name: str = None,
        **kwargs: Any,  # noqa
    ) -> np.ndarray:
        batch_size = 16 if "batch_size" not in kwargs else kwargs.pop("batch_size")
        all_embeddings = []
        for i in tqdm.tqdm(range(0, len(sentences), batch_size)):
            batch_texts = sentences[i : i + batch_size]

            batch_dict = self.create_batch_dict(self.tokenizer, batch_texts)
            batch_dict = {
                key: value.to(self.model.device) for key, value in batch_dict.items()
            }

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                    last_hidden_state = outputs.last_hidden_state
                    sequence_lengths = batch_dict["attention_mask"].sum(dim=1) - 1
                    batch_size = last_hidden_state.shape[0]
                    reps = last_hidden_state[
                        torch.arange(batch_size, device=last_hidden_state.device),
                        sequence_lengths,
                    ]
                    embeddings = F.normalize(reps, p=2, dim=-1)
                    all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list[str]] | list[str],
        prompt_name: str = None,
        **kwargs: Any,
    ) -> np.ndarray:
        sentences = corpus_to_texts(corpus, sep=" ")
        if "request_qid" in kwargs:
            kwargs.pop("request_qid")
        # NOTE: two spaces after the colon
        sentences = [f"passage:  {sentence}".strip() for sentence in sentences]
        print(f"Encoding corpus of length {len(sentences)}")
        print(f"First sentence: {sentences[0]}")
        return self.encode(sentences, **kwargs)

    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
        # NOTE: two spaces after the colon
        queries = [f"query:  {query.strip()}".strip() for query in queries]
        print(f"Encoding queries of length {len(queries)}")
        print(queries[0])
        return self.encode(queries, **kwargs)


def _loader(wrapper: type[RepLLaMAWrapper], **kwargs) -> Callable[..., Encoder]:
    _kwargs = kwargs

    def loader_inner(**kwargs: Any) -> Encoder:
        return wrapper(**_kwargs, **kwargs)

    return loader_inner


repllama_llama2_original = ModelMeta(
    loader=_loader(
        RepLLaMAWrapper,
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        peft_model_name_or_path="castorini/repllama-v1-7b-lora-passage",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="castorini/repllama-v1-7b-lora-passage",
    languages=["eng_Latn"],
    open_source=True,
    revision="01c7f73d771dfac7d292323805ebc428287df4f9-6097554dfe6e7d93e92f55010b678bcca1e233a8",  # base-peft revision
    release_date="2023-10-11",
)


repllama_llama2_reproduced = ModelMeta(
    loader=_loader(
        RepLLaMAWrapper,
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        peft_model_name_or_path="samaya-ai/RepLLaMA-reproduced",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="samaya-ai/RepLLaMA-reproduced",
    languages=["eng_Latn"],
    open_source=True,
    revision="01c7f73d771dfac7d292323805ebc428287df4f9-ad5c1d0938a1e02954bcafb4d811ba2f34052e71",  # base-peft revision
    release_date="2024-09-15",
)


## Debug code
# import mteb
# model = mteb.get_model("samaya-ai/RepLLaMA-reproduced")
# tasks = mteb.get_tasks(tasks=["SciFact"], languages=["eng"])
# evaluation = mteb.MTEB(tasks=tasks)
# evaluation.run(model)

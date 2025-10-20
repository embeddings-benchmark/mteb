import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb._requires_package import requires_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.models_protocols import EncoderProtocol
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class RepLLaMAModel(AbsEncoder):
    def __init__(
        self,
        peft_model_name_or_path: str,
        *,
        base_model_name_or_path: str,
        torch_dtype: torch.dtype,
        device_map: str,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ):
        from transformers import AutoModel, AutoTokenizer

        requires_package(
            self, "peft", peft_model_name_or_path, "pip install 'mteb[peft]'"
        )
        from peft import PeftModel

        self.base_model = AutoModel.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model = PeftModel.from_pretrained(self.base_model, peft_model_name_or_path)
        self.model = self.model.merge_and_unload()

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        # set the max_length for the evals as they did, although the model can handle longer
        self.model.config.max_length = 512
        self.tokenizer.model_max_length = 512
        self.model_prompts = self.validate_task_to_prompt_name(model_prompts)

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

    def combine_query_and_instruction(self, query, instruction):
        end_punct = "?" if query.strip()[-1] not in ["?", ".", "!"] else ""
        return f"{query}{end_punct} {instruction}".strip()

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        batch_size = 16 if "batch_size" not in kwargs else kwargs.pop("batch_size")
        all_embeddings = []
        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        prompt = self.model_prompts.get(prompt_name)
        sentences = [text for batch in inputs for text in batch["text"]]

        if prompt:
            if prompt_type == "queries":
                sentences = [
                    f"{prompt}{sentence.strip()}".strip() for sentence in sentences
                ]
            else:
                sentences = [f"{prompt}{sentence}".strip() for sentence in sentences]

        for i in tqdm(range(0, len(sentences), batch_size)):
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


def _loader(wrapper: type[RepLLaMAModel], **kwargs) -> Callable[..., EncoderProtocol]:
    _kwargs = kwargs

    def loader_inner(**kwargs: Any) -> EncoderProtocol:
        return wrapper(**_kwargs, **kwargs)

    return loader_inner


model_prompts = {
    PromptType.query.value: "query:  ",
    PromptType.document.value: "passage:  ",
}

REPLLAMA_CITATION = """
@article{rankllama,
      title={Fine-Tuning LLaMA for Multi-Stage Text Retrieval},
      author={Xueguang Ma and Liang Wang and Nan Yang and Furu Wei and Jimmy Lin},
      year={2023},
      journal={arXiv:2310.08319},
}
"""

repllama_llama2_original = ModelMeta(
    loader=RepLLaMAModel,  # type: ignore
    loader_kwargs=dict(
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        model_prompts=model_prompts,
    ),
    name="castorini/repllama-v1-7b-lora-passage",
    languages=["eng-Latn"],
    open_weights=True,
    revision="01c7f73d771dfac7d292323805ebc428287df4f9-6097554dfe6e7d93e92f55010b678bcca1e233a8",  # base-peft revision
    release_date="2023-10-11",
    training_datasets={
        # "Tevatron/msmarco-passage-aug",
        "mMARCO-NL",  # translation not trained on
    },
    n_parameters=7_000_000,
    memory_usage_mb=27,
    max_tokens=4096,
    embed_dim=4096,
    license="apache-2.0",
    reference="https://huggingface.co/samaya-ai/castorini/repllama-v1-7b-lora-passage",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Tevatron"],
    use_instructions=True,
    citation=REPLLAMA_CITATION,
    public_training_code=None,
    public_training_data=None,
)


repllama_llama2_reproduced = ModelMeta(
    loader=RepLLaMAModel,  # type: ignore
    loader_kwargs=dict(
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        model_prompts=model_prompts,
    ),
    name="samaya-ai/RepLLaMA-reproduced",
    languages=["eng-Latn"],
    open_weights=True,
    revision="01c7f73d771dfac7d292323805ebc428287df4f9-ad5c1d0938a1e02954bcafb4d811ba2f34052e71",  # base-peft revision
    release_date="2024-09-15",
    n_parameters=7_000_000,
    memory_usage_mb=27,
    max_tokens=4096,
    embed_dim=4096,
    license="apache-2.0",
    reference="https://huggingface.co/samaya-ai/RepLLaMA-reproduced",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Tevatron"],
    use_instructions=True,
    citation=REPLLAMA_CITATION,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

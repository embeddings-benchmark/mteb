from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from transformers import AutoModel, AutoTokenizer

from mteb.encoder_interface import Encoder, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)


class RepLLaMAWrapper(Wrapper):
    def __init__(
        self,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        torch_dtype: torch.dtype,
        device_map: str,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ):
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
        self.model_prompts = (
            self.validate_task_to_prompt_name(model_prompts) if model_prompts else None
        )

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
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,  # noqa
    ) -> np.ndarray:
        batch_size = 16 if "batch_size" not in kwargs else kwargs.pop("batch_size")
        all_embeddings = []
        prompt = self.get_prompt_name(self.model_prompts, task_name, prompt_type)
        if prompt:
            sentences = [f"{prompt}{sentence}".strip() for sentence in sentences]
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


def _loader(wrapper: type[RepLLaMAWrapper], **kwargs) -> Callable[..., Encoder]:
    _kwargs = kwargs

    def loader_inner(**kwargs: Any) -> Encoder:
        return wrapper(**_kwargs, **kwargs)

    return loader_inner


model_prompts = {
    PromptType.query.value: "query:  ",
    PromptType.passage.value: "passage:  ",
}

repllama_llama2_original = ModelMeta(
    loader=_loader(
        RepLLaMAWrapper,
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        peft_model_name_or_path="castorini/repllama-v1-7b-lora-passage",
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
        "Tevatron/msmarco-passage-aug": ["train"],
        "mMARCO-NL": ["train"],  # translation not trained on
    },
    n_parameters=7_000_000,
    memory_usage_mb=27,
    max_tokens=4096,
    embed_dim=4096,
    license="apache-2.0",
    reference="https://huggingface.co/samaya-ai/castorini/repllama-v1-7b-lora-passage",
    similarity_fn_name="cosine",
    framework=["PyTorch", "Tevatron"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
)


repllama_llama2_reproduced = ModelMeta(
    loader=_loader(
        RepLLaMAWrapper,
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        peft_model_name_or_path="samaya-ai/RepLLaMA-reproduced",
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
    similarity_fn_name="cosine",
    framework=["PyTorch", "Tevatron"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

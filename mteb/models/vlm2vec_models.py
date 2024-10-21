from __future__ import annotations

import logging
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
from peft import LoraConfig, PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts

from .instructions import task_to_instruction

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EncodeTypes = Literal["query", "passage"]


def llm2vec_instruction(instruction):
    if len(instruction) > 0 and instruction[-1] != ":":
        instruction = instruction.strip(".") + ":"
    return instruction


class VLM2VecWrapper:
    """Adapted from https://github.com/TIGER-AI-Lab/VLM2Vec/blob/main/src/model.py"""
    def __init__(self, model_name: str = "TIGER-Lab/VLM2Vec-LoRA", device: str = "cuda" if torch.cuda.is_available() else "cpu", **kwargs):
        try:
            import flash_attn  # noqa
        except ImportError:
            logger.warning(
                "VLM2Vec models were trained with flash attention enabled. For optimal performance, please install the `flash_attn` package with `pip install flash-attn --no-build-isolation`."
            )

        self.pooling = "last"
        self.normalize = True
        self.temperature = 1.0
        self.hidden_size = 4096
        self.device = device

        # Loading the base model
        base_model_name = "microsoft/Phi-3.5-vision-instruct"
        config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
        config.use_cache = False
        config.padding_side = "right"

        checkpoint_path = model_name if model_name else base_model_name
        base_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        base_model.padding_side = "right"

        # Building the model on top of the base
        if "LoRA" in model_name:
            lora_config = LoraConfig.from_pretrained(checkpoint_path)
            lora_model = PeftModel.from_pretrained(
                base_model, checkpoint_path, config=lora_config
            )
            lora_model = lora_model.merge_and_unload()
            model = lora_model
        else:
            model = base_model

        model.eval()
        self.mdl = model

        self.processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            num_crops=4,
        )

    def to(self, device: torch.device) -> None:
        self.mdl.to(device, dtype=torch.bfloat16)

    def encode(
        self,
        sentences: list[str],
        *,
        prompt_name: str = None,
        **kwargs: Any,  # noqa
    ) -> np.ndarray:
        if prompt_name is not None:
            instruction = (
                self.task_to_instructions[prompt_name]
                if self.task_to_instructions
                and prompt_name in self.task_to_instructions
                else llm2vec_instruction(task_to_instruction(prompt_name))
            )
        else:
            instruction = ""

        sentences = [[instruction, sentence] for sentence in sentences]
        return self.model.encode(sentences, **kwargs)

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list[str]] | list[str],
        prompt_name: str = None,
        **kwargs: Any,
    ) -> np.ndarray:
        sentences = corpus_to_texts(corpus, sep=" ")
        sentences = [["", sentence] for sentence in sentences]
        if "request_qid" in kwargs:
            kwargs.pop("request_qid")
        return self.model.encode(sentences, **kwargs)

    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
        return self.encode(queries, **kwargs)

    def encode_input(self, input):
        hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
        hidden_states = hidden_states.hidden_states[-1]
        pooled_output = self._pooling(hidden_states, input['attention_mask'])
        return pooled_output
    
    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == 'last':
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps
    

    def get_image_embeddings(
        self, images: list[Image.Image] | DataLoader, batch_size: int = 32
    ):
        all_image_embeddings = []
        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in tqdm(images):
                    inputs = self.processor(
                        images=batch, return_tensors="pt", padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_outputs = self.model.get_image_features(**inputs)
                    all_image_embeddings.append(image_outputs.cpu())
        else:
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    inputs = self.processor(
                        images=batch_images, return_tensors="pt", padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_outputs = self.model.get_image_features(**inputs)
                    all_image_embeddings.append(image_outputs.cpu())

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        return all_image_embeddings


    def get_text_embeddings(self, texts: list[str], batch_size: int = 32):
        all_text_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                inputs = self.processor(
                    text=batch_texts, return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_outputs = self.encode_input(**inputs)
                all_text_embeddings.append(text_outputs.cpu())

        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        return all_text_embeddings


    def get_fused_embeddings(
        self,
        texts: list[str] = None,
        images: list[Image.Image] | DataLoader = None,
        fusion_mode="sum",
        batch_size: int = 32,
    ):
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        text_embeddings = None
        image_embeddings = None

        if texts is not None:
            text_embeddings = self.get_text_embeddings(texts, batch_size)

        if images is not None:
            image_embeddings = self.get_image_embeddings(images, batch_size)

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            if fusion_mode == "sum":
                fused_embeddings = text_embeddings + image_embeddings
            else:
                # to do: add other fusion mode
                raise ValueError(f"fusion mode {fusion_mode} hasn't been implemented")
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings


vlm2vec_lora = ModelMeta(
    loader=partial(
        VLM2VecWrapper,
        model_name="TIGER-Lab/VLM2Vec-LoRA",
    ),
    name="TIGER-Lab/VLM2Vec-LoRA",
    languages=["eng_Latn"],
    open_source=True,
    revision="7403b6327958071c1e33c822c7453adadccc7298",
    release_date="2024-10-08",
)

vlm2vec_full = ModelMeta(
    loader=partial(
        VLM2VecWrapper,
        model_name="TIGER-Lab/VLM2Vec-Full",
    ),
    name="TIGER-Lab/VLM2Vec-Full",
    languages=["eng_Latn"],
    open_source=True,
    revision="e9afa98002097ac2471827ba23ea1f2ddd229480",
    release_date="2024-10-08",
)

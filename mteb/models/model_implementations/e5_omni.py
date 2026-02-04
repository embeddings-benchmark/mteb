from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb._requires_package import (
    requires_image_dependencies,
    requires_package,
)
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.model_implementations.bge_models import (
    bge_m3_training_data,
    bgem3_languages,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class E5OmniWrapper(AbsEncoder):
    """Wrapper for E5-Omni models."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype | str | None = torch.bfloat16,
        **kwargs: Any,
    ):
        requires_image_dependencies()
        requires_package(self, "transformers", model_name, "pip install mteb[e5-omni]")
        requires_package(self, "qwen_vl_utils", model_name, "pip install mteb[e5-omni]")
        from transformers import (
            AutoProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name,
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch_dtype,
            **kwargs,
        ).to(self.device)
        if hasattr(self.model, "padding_side"):
            self.model.padding_side = "left"
        self.model.eval()

    @torch.no_grad()
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
        all_embeddings = []

        for batch in tqdm(inputs, desc="Encoding"):
            batch_texts = batch.get("text", [])
            batch_images = batch.get("image", [])

            if not batch_texts and not batch_images:
                raise ValueError("No text or image features found in batch.")

            messages = []
            max_len = max(len(batch_texts), len(batch_images))
            for i in range(max_len):
                content = []
                if i < len(batch_texts):
                    content.append({"type": "text", "text": batch_texts[i]})
                if i < len(batch_images):
                    content.append({"type": "image", "image": batch_images[i]})
                messages.append([{"role": "user", "content": content}])

            texts = []
            for msg in messages:
                rendered = self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                if isinstance(rendered, list):
                    rendered = rendered[0]
                texts.append(f"{rendered}<|endoftext|>")

            image_inputs = None
            video_inputs = None
            audio_inputs = None
            if batch_images:
                from qwen_vl_utils import process_vision_info

                image_inputs, video_inputs = process_vision_info(messages)

            model_inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                audio=audio_inputs,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            # Prepare inputs for generation to handle cache_position and other requirements for Qwen2.5-Omni
            cache_position = torch.arange(
                0, model_inputs["input_ids"].shape[1], device=self.device
            )
            model_inputs = self.model.prepare_inputs_for_generation(
                **model_inputs, use_cache=True, cache_position=cache_position
            )

            outputs = self.model(**model_inputs, output_hidden_states=True)

            # For E5-Omni, we use the last hidden state of the last token
            # as is common for decoder-only LLM-based embedding models.
            last_hidden_state = outputs.hidden_states[-1]

            # Find the last non-padding token
            attention_mask = model_inputs["attention_mask"]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            embeddings = last_hidden_state[
                torch.arange(last_hidden_state.size(0)), sequence_lengths
            ]

            # Normalize embeddings as recommended by the authors
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

            all_embeddings.append(embeddings.cpu().to(torch.float32))

        return torch.cat(all_embeddings, dim=0).numpy()


E5_OMNI_CITATION = """@misc{chen2026e5omniexplicitcrossmodalalignment,
      title={e5-omni: Explicit Cross-modal Alignment for Omni-modal Embeddings}, 
      author={Haonan Chen and Sicheng Gao and Radu Timofte and Tetsuya Sakai and Zhicheng Dou},
      year={2026},
      eprint={2601.03666},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.03666}, 
}"""

E5_OMNI_TRAINING_DATASETS = bge_m3_training_data | {
    "MMEB-V1",
    "MMEB-V2",
    "PixMo-Docs",
    "MSR-VTT",
    "AudioCaps",
}

e5_omni_3b = ModelMeta(
    loader=E5OmniWrapper,
    name="Haon-Chen/e5-omni-3B",
    languages=bgem3_languages,
    revision="d2765489f361965142c069c2dc18291220a3819a",
    release_date="2026-01-07",
    modalities=[
        "text",
        "image",
    ],  # Wrapper currently supports text/image only.
    n_parameters=5_000_000_000,
    memory_usage_mb=8971,
    max_tokens=512,  # They use 512 in the training, despite the underlying model can handle more
    embed_dim=2048,
    license="mit",
    open_weights=True,
    framework=["PyTorch", "Transformers"],
    reference="https://arxiv.org/abs/2601.03666",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=E5_OMNI_TRAINING_DATASETS,
    public_training_code=None,
    public_training_data=None,
    citation=E5_OMNI_CITATION,
    adapted_from="Qwen/Qwen2.5-Omni-3B",
)

e5_omni_7b = ModelMeta(
    loader=E5OmniWrapper,
    name="Haon-Chen/e5-omni-7B",
    languages=bgem3_languages,
    revision="bbf5f87c0899abf7890bca98c307113f3c813041",
    release_date="2026-01-07",
    modalities=[
        "text",
        "image",
    ],  # Wrapper currently supports text/image only.
    n_parameters=9_000_000_000,
    memory_usage_mb=17036,
    max_tokens=512,  # They use 512 in the training, despite the underlying model can handle more
    embed_dim=3584,
    license="mit",
    open_weights=True,
    framework=["PyTorch", "Transformers"],
    reference="https://arxiv.org/abs/2601.03666",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=E5_OMNI_TRAINING_DATASETS,
    public_training_code=None,
    public_training_data=None,
    citation=E5_OMNI_CITATION,
    adapted_from="Qwen/Qwen2.5-Omni-7B",
)

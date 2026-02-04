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
from mteb.models.model_implementations.bge_models import (
    bge_m3_training_data,
    bgem3_languages,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

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
        requires_package(
            self, "qwen_omni_utils", model_name, "pip install mteb[e5-omni]"
        )
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

        # Map finetuned model names to their base processor models
        # The processor should come from the base Qwen model, not the finetuned repo
        processor_model_map = {
            "Haon-Chen/e5-omni-3B": "Qwen/Qwen2.5-Omni-3B",
            "Haon-Chen/e5-omni-7B": "Qwen/Qwen2.5-Omni-7B",
        }
        processor_model = processor_model_map.get(model_name, model_name)

        self.processor = AutoProcessor.from_pretrained(
            processor_model,
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

    @staticmethod
    def _to_text(x: Any) -> str:
        """Normalize text input: handle dicts (title/text) or return string."""
        if isinstance(x, dict):
            title = x.get("title", "") or ""
            text = x.get("text", "") or x.get("body", "") or ""
            if title and text:
                return f"{title}\n{text}"
            return title or text
        return "" if x is None else str(x)

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
            # Normalize text inputs: handle dicts (title/text) from BEIR-style corpora
            raw_texts = batch.get("text", [])
            batch_texts = [self._to_text(x) for x in raw_texts]
            batch_images = batch.get("image", [])

            if not batch_texts and not batch_images:
                raise ValueError("No text or image features found in batch.")

            if prompt_type == PromptType.query:
                text_prefix = "Query: "
            else:
                text_prefix = ""

            messages = []
            max_len = max(len(batch_texts), len(batch_images))
            for i in range(max_len):
                content = []
                if i < len(batch_texts):
                    # Prepend the appropriate prefix to text
                    prefixed_text = (
                        f"{text_prefix}{batch_texts[i]}"
                        if text_prefix
                        else batch_texts[i]
                    )
                    content.append({"type": "text", "text": prefixed_text})
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

            from qwen_omni_utils import process_mm_info

            audios, images, videos = [], [], []
            for msg in messages:
                a, im, v = process_mm_info(msg, use_audio_in_video=True)
                audios.append(a)
                images.append(im)
                videos.append(v)

            # Check if we have any actual multimodal content (not all None)
            # If all are None, pass None to processor instead of a list of Nones
            has_audio = any(a is not None for a in audios)
            has_images = any(im is not None for im in images)
            has_videos = any(v is not None for v in videos)

            audio_inputs = audios if has_audio else None
            image_inputs = images if has_images else None
            video_inputs = videos if has_videos else None

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

            # Run a plain forward pass and pool exactly as in the model card:
            # last_hidden_state[:, -1] with left padding.
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_hidden_states=True,
                use_cache=False,
            )
            last_hidden_state = outputs.hidden_states[-1]
            embeddings = last_hidden_state[:, -1]

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
    n_parameters=4_703_464_448,
    n_embedding_parameters=311_164_928,
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
    n_parameters=8_931_813_888,
    n_embedding_parameters=544_997_376,
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

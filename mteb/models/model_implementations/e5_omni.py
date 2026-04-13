from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb._create_dataloaders import AudioCollator
from mteb._requires_package import (
    requires_audio_dependencies,
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
    from typing_extensions import Unpack

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs

logger = logging.getLogger(__name__)


class E5OmniWrapper(AbsEncoder):
    """Wrapper for E5-Omni models."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype | str | None = None,
        max_audio_length_seconds: float | None = None,
        **kwargs: Any,
    ):
        requires_image_dependencies()
        requires_audio_dependencies()
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

        processor_model_map = {
            "Haon-Chen/e5-omni-3B": "Qwen/Qwen2.5-Omni-3B",
            "Haon-Chen/e5-omni-7B": "Qwen/Qwen2.5-Omni-7B",
        }
        processor_model = processor_model_map.get(model_name, model_name)

        self.processor = AutoProcessor.from_pretrained(
            processor_model,
            use_fast=False,
        )
        self.processor.tokenizer.padding_side = "left"

        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch_dtype,
            **kwargs,
        ).to(self.device)
        self.model.padding_side = "left"
        self.model.eval()

        self.sampling_rate = self.processor.feature_extractor.sampling_rate
        self.max_samples = int((max_audio_length_seconds or 30.0) * self.sampling_rate)

    @torch.no_grad()
    def encode(  # noqa: PLR0914
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        from qwen_omni_utils import process_mm_info

        all_embeddings = []

        for batch in tqdm(inputs, desc="Encoding"):
            batch_texts = batch.get("text", [])
            batch_images = batch.get("image", [])
            batch_audios = batch.get("audio", [])

            text_prefix = "Query: " if prompt_type == PromptType.query else ""
            max_len = max(len(batch_texts), len(batch_images), len(batch_audios))

            for i in range(max_len):
                # Build a single conversation message
                content = []
                if i < len(batch_texts):
                    t = (
                        f"{text_prefix}{batch_texts[i]}"
                        if text_prefix
                        else batch_texts[i]
                    )
                    content.append({"type": "text", "text": t})
                if i < len(batch_images):
                    content.append({"type": "image", "image": batch_images[i]})
                if i < len(batch_audios):
                    audio_array = AudioCollator.resample_audio(
                        {"audio": batch_audios[i]},
                        self.sampling_rate,
                        self.max_samples,
                    )
                    content.append({"type": "audio", "audio": audio_array})

                message = [{"role": "user", "content": content}]

                # Process exactly like the model card
                text = self.processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                if isinstance(text, list):
                    text = text[0]
                text = f"{text}<|endoftext|>"

                audios, images, videos = process_mm_info(
                    message, use_audio_in_video=True
                )

                is_mm = bool(audios or images or videos)
                model_inputs = self.processor(
                    text=text,
                    images=images if images else None,
                    videos=videos if videos else None,
                    audio=audios if audios else None,
                    padding="longest",
                    return_tensors="pt",
                    max_length=None if is_mm else 512,
                    truncation=not is_mm,
                ).to(self.device)

                cache_position = torch.arange(
                    0, model_inputs["input_ids"].shape[1], device=self.device
                )
                prepared = self.model.prepare_inputs_for_generation(
                    **model_inputs, use_cache=True, cache_position=cache_position
                )
                outputs = self.model(
                    **prepared, return_dict=True, output_hidden_states=True
                )
                emb = outputs.hidden_states[-1][:, -1]
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)


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
    loader_kwargs=dict(
        torch_dtype=torch.bfloat16,
        max_audio_length_seconds=30.0,
    ),
    name="Haon-Chen/e5-omni-3B",
    languages=bgem3_languages,
    revision="e3530921a79351089af9376317b95d70470eabd6",
    release_date="2026-01-09",
    modalities=[
        "text",
        "image",
        "audio",
    ],
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
    loader_kwargs=dict(
        torch_dtype=torch.bfloat16,
        max_audio_length_seconds=30.0,
    ),
    name="Haon-Chen/e5-omni-7B",
    languages=bgem3_languages,
    revision="0514b08201bc73b9a6025b1a05815ff69334806c",
    release_date="2026-01-09",
    modalities=[
        "text",
        "image",
        "audio",
    ],
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

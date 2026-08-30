from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import FramesCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs

_IMAGE_DOCUMENT_PROMPT = "Find an image caption describing the given image."
_QUERY_DEFAULT_PROMPT = "Represent the given input for retrieval."
_VIDEO_DOCUMENT_PROMPT = "Describe this video in detail."

_UNIME_V2_CITATION = r"""
@misc{gu2025unimev2mllmasajudgeuniversalmultimodal,
  title={UniME-V2: MLLM-as-a-Judge for Universal Multimodal Embedding Learning},
  author={Tiancheng Gu and Kaicheng Yang and Kaichen Zhang and Xiang An and Ziyong Feng and Yueyi Zhang and Weidong Cai and Jiankang Deng and Lidong Bing},
  year={2025},
  eprint={2510.13515},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2510.13515},
}
"""

_UNIME_V2_TRAINING_DATASETS = {
    "CIRRIT2IRetrieval",
    "FashionIQIT2IRetrieval",
    "HatefulMemesI2TRetrieval",
    "HatefulMemesT2IRetrieval",
    "Imagenet1k",
    "MSCOCOI2TRetrieval",
    "MSCOCOT2IRetrieval",
    "NIGHTSI2IRetrieval",
    "OKVQAIT2TRetrieval",
    "OVENIT2ITRetrieval",
    "OVENIT2TRetrieval",
    "SUN397",
    "SUN397ZeroShot",
    "VOC2007",
    "VisualNewsI2TRetrieval",
    "VisualNewsT2IRetrieval",
    "VizWizIT2TRetrieval",
    "WebQAT2ITRetrieval",
    "WebQAT2TRetrieval",
}


class UniMEV2Wrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        fps: float | None = 2.0,
        max_frames: int | None = 64,
        num_frames: int | None = None,
        attn_implementation: str = "sdpa",
        use_task_instructions: bool = True,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.use_task_instructions = use_task_instructions

        dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", torch.bfloat16))
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
        )
        self.processor.tokenizer.padding_side = "left"

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            revision=revision,
            dtype=dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )
        self.model.config.use_cache = False
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _pooling(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        batch_size = last_hidden_state.shape[0]
        if left_padding:
            embeddings = last_hidden_state[torch.arange(batch_size), -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            embeddings = last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                sequence_lengths,
            ]
        return torch.nn.functional.normalize(embeddings, p=2, dim=-1)

    @staticmethod
    def _build_conversation(
        *,
        text: str | None,
        image: Any | None,
        video: Any | None,
        instruction: str,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        if video is not None:
            content.append({"type": "video", "video": video})

        prompt_parts = [
            part.strip() for part in (instruction, text or "") if part.strip()
        ]
        prompt = "\n".join(prompt_parts)
        if not prompt:
            if image is not None:
                prompt = _IMAGE_DOCUMENT_PROMPT
            elif video is not None:
                prompt = _VIDEO_DOCUMENT_PROMPT
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _conversation_batches(
        conversations: list[list[dict[str, Any]]], *, has_video: bool
    ) -> list[list[list[dict[str, Any]]]]:
        if has_video:
            return [[conversation] for conversation in conversations]
        return [conversations]

    def _get_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str:
        if not self.use_task_instructions:
            return ""
        prompt = task_metadata.prompt
        if isinstance(prompt, dict) and prompt_type:
            return prompt.get(prompt_type.value, "").strip()
        if prompt_type == PromptType.query:
            return prompt.strip() if isinstance(prompt, str) else _QUERY_DEFAULT_PROMPT
        return ""

    @torch.inference_mode()
    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        features = inputs.dataset.features
        has_text = "text" in features
        has_image = "image" in features
        has_video = "video" in features

        if has_video:
            inputs.collate_fn = FramesCollator(
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
            )

        instruction = self._get_instruction(task_metadata, prompt_type)
        show_progress_bar = kwargs.get("show_progress_bar", True)
        all_embeddings: list[torch.Tensor] = []

        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding"):
            batch_size = len(next(iter(batch.values())))
            conversations = [
                self._build_conversation(
                    text=batch["text"][i] if has_text else None,
                    image=batch["image"][i] if has_image else None,
                    video=batch["video"][i] if has_video else None,
                    instruction=instruction,
                )
                for i in range(batch_size)
            ]
            for conversation_batch in self._conversation_batches(
                conversations, has_video=has_video
            ):
                model_inputs = self.processor.apply_chat_template(
                    conversation_batch,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs={"padding": True},
                ).to(self.device)
                output = self.model(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
                embeddings = self._pooling(
                    output.hidden_states[-1], model_inputs["attention_mask"]
                )
                all_embeddings.append(embeddings.cpu().to(torch.float32))

        return torch.cat(all_embeddings, dim=0)


unime_v2_llava_onevision_8b = ModelMeta(
    loader=UniMEV2Wrapper,
    name="TianchengGu/UniME-V2-LLaVA-OneVision-8B",
    revision="36ef54da9f3dc3a2bfba115c4fb403b1a1f7cb0c",
    release_date="2025-10-15",
    languages=["eng-Latn"],
    n_parameters=8_030_807_584,
    n_embedding_parameters=545_226_752,
    memory_usage_mb=15_318,
    max_tokens=32_768,
    embed_dim=3_584,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/GaryGuTC/UniME-v2",
    public_training_data="https://huggingface.co/datasets/TIGER-Lab/MMEB-train",
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/TianchengGu/UniME-V2-LLaVA-OneVision-8B",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=_UNIME_V2_TRAINING_DATASETS,
    adapted_from="llava-hf/llava-onevision-qwen2-7b-ov-hf",
    modalities=["text", "image", "video"],
    model_type=["dense"],
    citation=_UNIME_V2_CITATION,
    extra_requirements_groups=["transformers-v5"],
)

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

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

GVE_CITATION = """@misc{guo2025gve,
  title={Towards Universal Video Retrieval: Generalizing Video Embedding via Synthesized Multimodal Pyramid Curriculum},
  author={Zhuoning Guo and Mingxin Li and Yanzhao Zhang and Dingkun Long and Pengjun Xie and Xiaowen Chu},
  year={2025},
  eprint={2510.27571},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2510.27571}
}"""

_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
_VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
_DEFAULT_INSTRUCTION = "You are a helpful assistant."


class GVEWrapper(AbsEncoder):
    """Wrapper for Alibaba-NLP GVE (General Video Embedder) models.

    GVE is built on Qwen2.5-VL and embeds text, images, videos, and their
    combinations. The HF repos ship a custom `Qwen25VLForEmbedding` class,
    but it is a plain subclass of `Qwen2_5_VLForConditionalGeneration` with
    no extra weights that returns hidden states instead of logits (and its
    remote code is incompatible with transformers >= 4.56), so the native
    class is loaded instead and the final hidden states are read from
    `output_hidden_states`. Embeddings are the L2-normalized last-token
    hidden state with left padding, following the model card.

    Frame sampling defaults to fps=2 capped at 32 frames, denser than the
    model card's 8-frame demo: with the 200*28*28 per-frame pixel budget,
    max_length is 4096 (vs the demo's 1200) so dense video batches never
    truncate vision tokens; text-only batches still pad to longest-in-batch.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        *,
        device: str | None = None,
        max_length: int = 4096,
        fps: float | None = 2.0,
        max_frames: int | None = 32,
        num_frames: int | None = None,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.max_length = max_length
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            revision=revision,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        self.model.eval()
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            model_name, revision=revision, use_fast=True
        )
        self.processor.tokenizer.padding_side = "left"
        video_processor = getattr(self.processor, "video_processor", None)
        if video_processor is not None:
            # matches the model card's video settings; frame sampling is
            # already done by FramesCollator. Newer processors read the
            # pixel budget from size["longest_edge"], older ones from
            # max_pixels; set both so frames stay within max_length.
            max_video_pixels = 200 * 28 * 28
            if isinstance(getattr(video_processor, "size", None), dict):
                video_processor.size = {
                    **video_processor.size,
                    "longest_edge": max_video_pixels,
                }
            if hasattr(video_processor, "max_pixels"):
                video_processor.max_pixels = max_video_pixels
            if hasattr(video_processor, "do_sample_frames"):
                video_processor.do_sample_frames = False

    @staticmethod
    def _build_prompt(content: str, instruction: str | None) -> str:
        system = instruction or _DEFAULT_INSTRUCTION
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @torch.inference_mode()
    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        instruction = None
        if prompt_type != PromptType.document:
            instruction = self.get_instruction(task_metadata, prompt_type)

        if "video" in inputs.dataset.features:
            inputs.collate_fn = FramesCollator(
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
            )

        all_embeddings = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding"):
            batch_size = len(next(iter(batch.values())))
            texts = batch.get("text", [None] * batch_size)
            images = batch.get("image", [None] * batch_size)
            videos = batch.get("video", [None] * batch_size)

            prompts = []
            for text, image, video in zip(texts, images, videos):
                content = ""
                if video is not None:
                    content += _VIDEO_PLACEHOLDER
                if image is not None:
                    content += _IMAGE_PLACEHOLDER
                if text is not None:
                    content += text
                prompts.append(self._build_prompt(content, instruction))

            image_inputs = [img.convert("RGB") for img in images if img is not None]
            video_inputs = [vid for vid in videos if vid is not None]
            processed = self.processor(
                text=prompts,
                images=image_inputs or None,
                videos=video_inputs or None,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            processed = processed.to(self.device)

            # logits_to_keep=1 skips computing vocab logits we don't use
            outputs = self.model(
                **processed, output_hidden_states=True, logits_to_keep=1
            )
            # left padding -> last position is the last real token
            embeddings = torch.nn.functional.normalize(
                outputs.hidden_states[-1][:, -1, :], p=2, dim=1
            )
            all_embeddings.append(embeddings.float().cpu())
        return torch.cat(all_embeddings, dim=0).numpy()


gve_training_datasets = set(
    # WebVid-10M
    # InternVid
    # VAST-2M-Vi
    # synthesized multimodal pyramid curriculum data (13M pairs)
)

gve_3b = ModelMeta(
    loader=GVEWrapper,
    name="Alibaba-NLP/GVE-3B",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="2c2962d4bb1503b478b8a0d975d7b9fdcefbd3f3",
    release_date="2025-10-31",
    modalities=["image", "text", "video"],
    n_parameters=3754622976,
    n_embedding_parameters=311164928,
    memory_usage_mb=7161,
    max_tokens=32768,
    embed_dim=2048,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Alibaba-NLP/GVE-3B",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=gve_training_datasets,
    adapted_from="Qwen/Qwen2.5-VL-3B-Instruct",
    citation=GVE_CITATION,
)

gve_7b = ModelMeta(
    loader=GVEWrapper,
    name="Alibaba-NLP/GVE-7B",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="3046045eeed08e1d8731c3b943827b81b2b20817",
    release_date="2025-10-31",
    modalities=["image", "text", "video"],
    n_parameters=8292166656,
    n_embedding_parameters=544997376,
    memory_usage_mb=15816,
    max_tokens=32768,
    embed_dim=3584,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Alibaba-NLP/GVE-7B",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=gve_training_datasets,
    adapted_from="Qwen/Qwen2.5-VL-7B-Instruct",
    citation=GVE_CITATION,
)

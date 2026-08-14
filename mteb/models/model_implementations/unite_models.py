from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from tqdm.autonotebook import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import VideoCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

UNITE_CITATION = """@article{kong2025modality,
  title={Modality Curation: Building Universal Embeddings for Advanced Multimodal Information Retrieval},
  author={Kong, Fanheng and Zhang, Jingyuan and Liu, Yahui and Zhang, Hongzhi and Feng, Shi and Yang, Xiaocui and Wang, Daling and Tian, Yu and W., Victoria and Zhang, Fuzheng and Zhou, Guorui},
  journal={arXiv preprint arXiv:2505.19650},
  year={2025}
}"""

# Vendored from https://github.com/friedrichor/UNITE/blob/main/inference/modeling_unite.py
# The HF repos ship no auto_map, so trust_remote_code is not available.
# Two lines differ from upstream, both required by the transformers >=4.52
# Qwen2VL refactor; each is marked inline.


def _fit(frame, max_pixels: int):
    """Downscale a video frame to at most max_pixels, keeping aspect ratio.

    UNITE's reference inference passes video through qwen_vl_utils with
    max_pixels=360*420, far below the image budget. Without this, video items
    carry ~6x the visual tokens they should and encoding is ~6x slower.
    """
    w, h = frame.size
    if w * h <= max_pixels:
        return frame
    scale = (max_pixels / (w * h)) ** 0.5
    nw = max(28, int(w * scale) // 28 * 28)
    nh = max(28, int(h * scale) // 28 * 28)
    return frame.resize((nw, nh))


def _unwrap(out):
    """tf 4.52 vision tower returns a tensor; tf 5.x returns an output object."""
    return out.pooler_output if hasattr(out, "pooler_output") else out


def _load_base(model_name: str, revision: str | None):
    from transformers import Qwen2VLForConditionalGeneration

    # UNITE ships modeling_unite.py as a Qwen2VLForConditionalGeneration subclass,
    # but subclassing breaks checkpoint key remapping on transformers 5.x: the
    # language model silently loads with random weights. We keep the base class and
    # reproduce UNITE's pooling in _unite_embed instead.
    try:
        return Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, revision=revision, dtype=torch.bfloat16
        )
    except TypeError:
        return Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, revision=revision, torch_dtype=torch.bfloat16
        )


def _scatter(model, inputs_embeds, input_ids, pixels, grid, token_id):
    tower = model.model.visual
    embeds = _unwrap(tower(pixels.type(tower.dtype), grid_thw=grid))
    mask = (input_ids == token_id).unsqueeze(-1).expand_as(inputs_embeds)
    return inputs_embeds.masked_scatter(
        mask.to(inputs_embeds.device),
        embeds.to(inputs_embeds.device, inputs_embeds.dtype),
    )


def _unite_embed(model, inputs) -> torch.Tensor:
    """UNITE's forward: embed, scatter visual features, decode, last-token pool."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    inputs_embeds = model.get_input_embeddings()(input_ids)

    if inputs.get("pixel_values") is not None:
        inputs_embeds = _scatter(
            model, inputs_embeds, input_ids, inputs["pixel_values"],
            inputs["image_grid_thw"], model.config.image_token_id,
        )
    if inputs.get("pixel_values_videos") is not None:
        inputs_embeds = _scatter(
            model, inputs_embeds, input_ids, inputs["pixel_values_videos"],
            inputs["video_grid_thw"], model.config.video_token_id,
        )

    out = model.model.language_model(
        input_ids=None,
        attention_mask=attention_mask.to(inputs_embeds.device),
        inputs_embeds=inputs_embeds,
    )
    h = out.last_hidden_state
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        emb = h[:, -1]
    else:
        idx = attention_mask.sum(dim=1) - 1
        emb = h[torch.arange(h.shape[0], device=h.device), idx]
    return torch.nn.functional.normalize(emb, p=2, dim=1).contiguous()


class UniteWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        min_image_tokens: int = 256,
        max_image_tokens: int = 1280,
        fps: float = 1.0,
        max_frames: int = 32,
        num_frames: int = 32,
        target_sampling_rate: int = 16000,
        video_max_pixels: int = 360 * 420,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.target_sampling_rate = target_sampling_rate
        self.video_max_pixels = video_max_pixels

        self.model = _load_base(model_name, revision)
        self.model.eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
            min_pixels=min_image_tokens * 28 * 28,
            max_pixels=max_image_tokens * 28 * 28,
        )

    @staticmethod
    def _suffix(has_text: bool, has_image: bool, has_video: bool) -> str:
        if has_video and has_text:
            return "\nSummary above sentence and video in one word:"
        if has_video:
            return "\nSummary above video in one word:"
        if has_image and has_text:
            return "\nSummary above sentence and image in one word:"
        if has_image:
            return "\nSummary above image in one word:"
        return "\nSummary above sentence in one word:"

    def _embed_one(self, text=None, image=None, video=None) -> torch.Tensor:
        content = []
        if video is not None:
            video = [_fit(f, self.video_max_pixels) for f in video]
            content.append({"type": "video", "video": ""})
        elif image is not None:
            content.append({"type": "image", "image": ""})
        if text:
            content.append({"type": "text", "text": text})
        content.append(
            {
                "type": "text",
                "text": self._suffix(bool(text), image is not None, video is not None),
            }
        )
        messages = [{"role": "user", "content": content}]
        prompt = (
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            + "<|endoftext|>"
        )
        inputs = self.processor(
            text=[prompt],
            images=[image] if image is not None else None,
            videos=[video] if video is not None else None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            return _unite_embed(self.model, inputs)

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
        if "video" in inputs.dataset.features:
            inputs.collate_fn = VideoCollator(
                target_sampling_rate=self.target_sampling_rate,
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
            )

        all_embeddings = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="UNITE Encoding"):
            texts = batch.get("text")
            images = batch.get("image")
            videos = batch.get("video")
            n = len(texts or images or videos)
            for i in range(n):
                emb = self._embed_one(
                    text=texts[i] if texts is not None else None,
                    image=images[i] if images is not None else None,
                    video=videos[i] if videos is not None else None,
                )
                all_embeddings.append(emb.float().cpu())
        return torch.cat(all_embeddings, dim=0)


unite_training_datasets = set(
    # Stage 1, retrieval adaptation: friedrichor/Unite-Base-Retrieval-Train
    # (includes Tarsier2-Recap-585K video captions)
    # Stage 2, instruction tuning: TIGER-Lab/MMEB-train
    # Following the VLM2Vec precedent, MMEB-train components are not enumerated
    # here even though mteb ships tasks matching several of them.
)


unite_base_qwen2vl_2b = ModelMeta(
    loader=UniteWrapper,
    name="friedrichor/Unite-Base-Qwen2-VL-2B",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="850056e7a4b4129768e1c76795eac3c4331de76d",
    release_date="2025-05-28",
    modalities=["image", "text", "video"],
    n_parameters=2_208_985_600,
    n_embedding_parameters=233_373_696,
    memory_usage_mb=8427,
    embed_dim=1536,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/friedrichor/Unite-Base-Qwen2-VL-2B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "safetensors", "Transformers"],
    use_instructions=False,
    public_training_code="https://github.com/friedrichor/UNITE",
    public_training_data="https://huggingface.co/datasets/friedrichor/Unite-Base-Retrieval-Train",
    training_datasets=unite_training_datasets,
    adapted_from="Qwen/Qwen2-VL-2B-Instruct",
    citation=UNITE_CITATION,
    contacts=["hubielu"],
)

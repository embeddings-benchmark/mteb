from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import FramesCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


def _proj(output, attr: str) -> torch.Tensor:
    """Pull the projected embedding off a Cosmos-Embed1 output object."""
    value = getattr(output, attr, None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"Expected a tensor at {type(output).__name__}.{attr}, got {type(value).__name__}"
        )
    return value


class CosmosEmbed1Model(AbsEncoder):
    """Wrapper for NVIDIA Cosmos-Embed1 joint video-text embedders."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_frames: int = 8,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        self.model_name = model_name
        self.device = device
        self.num_frames = num_frames
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
        )

    def _move(self, batch: Any) -> dict[str, Any]:
        return {
            key: (value.to(self.device) if hasattr(value, "to") else value)
            for key, value in dict(batch).items()
        }

    @torch.no_grad()
    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        all_embeddings = []
        for batch in tqdm(texts, disable=not show_progress_bar, desc="Text Encoding"):
            inputs = self._move(self.processor(text=batch["text"], return_tensors="pt"))
            output = self.model.get_text_embeddings(**inputs)
            embeddings = _proj(output, "text_proj")
            all_embeddings.append(embeddings.float().cpu())
        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def get_video_embeddings(
        self,
        videos: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        all_embeddings = []
        for batch in tqdm(videos, disable=not show_progress_bar, desc="Video Encoding"):
            # The collator already hands us [T, C, H, W] uint8 frames, which
            # is the BTCHW layout the processor wants once stacked. It requires
            # exactly num_frames frames per clip.
            frames = torch.stack(
                [
                    video.to(torch.uint8)
                    if isinstance(video, torch.Tensor)
                    else torch.as_tensor(video, dtype=torch.uint8)
                    for video in batch["video"]
                ]
            )
            inputs = self._move(self.processor(videos=frames, return_tensors="pt"))
            output = self.model.get_video_embeddings(**inputs)
            embeddings = _proj(output, "visual_proj")
            all_embeddings.append(embeddings.float().cpu())
        return torch.cat(all_embeddings, dim=0)

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
        has_text = "text" in inputs.dataset.features
        has_video = "video" in inputs.dataset.features

        if has_video:
            inputs.collate_fn = FramesCollator(num_frames=self.num_frames)

        text_embeddings = None
        video_embeddings = None

        if has_text:
            text_embeddings = self.get_text_embeddings(inputs, **kwargs)
        if has_video:
            video_embeddings = self.get_video_embeddings(inputs, **kwargs)

        if text_embeddings is not None and video_embeddings is not None:
            if len(text_embeddings) != len(video_embeddings):
                raise ValueError(
                    "The number of texts and videos must have the same length"
                )
            return text_embeddings + video_embeddings
        if text_embeddings is not None:
            return text_embeddings
        if video_embeddings is not None:
            return video_embeddings

        raise ValueError(
            f"No supported modality found in dataset features: {list(inputs.dataset.features.keys())}"
        )


_COSMOS_COMMON = dict(
    loader=CosmosEmbed1Model,
    model_type=["dense"],
    languages=["eng-Latn"],
    release_date="2025-05-26",
    modalities=["video", "text"],
    license="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers"],
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    max_tokens=128,
    training_datasets={
        "Kinetics400V",
        "Kinetics400VA",
        "Kinetics400ZeroShot",
        "Kinetics400VAZeroShot",
        "Kinetics600V",
        "Kinetics600VA",
        "Kinetics600VZeroShot",
        "Kinetics600VAZeroShot",
        "Kinetics700V",
        "Kinetics700VA",
        "Kinetics700VZeroShot",
        "Kinetics700VAZeroShot",
    },
    citation=None,
    loader_kwargs=dict(num_frames=8),
    extra_requirements_groups=["cosmos-embed1"],
)

cosmos_embed1_224p = ModelMeta(
    name="nvidia/Cosmos-Embed1-224p",
    revision="787e0b996f5260a71ad474a283c90539a2e12986",
    n_parameters=1_172_696_698,
    n_embedding_parameters=23_834_880,
    memory_usage_mb=4473,
    embed_dim=256,
    reference="https://huggingface.co/nvidia/Cosmos-Embed1-224p",
    **_COSMOS_COMMON,
)

cosmos_embed1_336p = ModelMeta(
    name="nvidia/Cosmos-Embed1-336p",
    revision="0e8a28f7bf370f2dcb3b9b61d23e167d0d6b0e6f",
    n_parameters=1_173_934_714,
    n_embedding_parameters=23_834_880,
    memory_usage_mb=4478,
    embed_dim=768,
    reference="https://huggingface.co/nvidia/Cosmos-Embed1-336p",
    **_COSMOS_COMMON,
)

cosmos_embed1_448p = ModelMeta(
    name="nvidia/Cosmos-Embed1-448p",
    revision="f60ec73636eb7c9cc25267367713b7b1b0cffaf3",
    n_parameters=1_174_565_498,
    n_embedding_parameters=23_834_880,
    memory_usage_mb=4481,
    embed_dim=768,
    reference="https://huggingface.co/nvidia/Cosmos-Embed1-448p",
    **_COSMOS_COMMON,
)

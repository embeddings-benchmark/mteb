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

_VICLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_VICLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_VICLIP_SIZE = 224

VICLIP_CITATION = """@article{wang2023internvid,
  title={InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation},
  author={Wang, Yi and He, Yinan and Li, Yizhuo and Li, Kunchang and Yu, Jiashuo and Ma, Xin and
          Li, Xinhao and Chen, Guo and Chen, Xinyuan and Wang, Yaohui and others},
  journal={arXiv preprint arXiv:2307.06942},
  year={2023}
}"""


class ViCLIPWrapper(AbsEncoder):
    """Wrapper for OpenGVLab ViCLIP models (ViCLIP-L-14-hf, ViCLIP-B-16-hf).

    ViCLIP extends CLIP's ViT backbone with spatiotemporal attention for video-text
    contrastive learning. It uses a custom BPE tokenizer (accessed via model.tokenizer)
    and non-standard encoding methods (get_vid_features / get_text_features).
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_frames: int = 8,
        **kwargs: Any,
    ):
        from transformers import AutoModel

        self.model_name = model_name
        self.device = device
        self.num_frames = num_frames
        self.model = AutoModel.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self.tokenizer = self.model.tokenizer

    def _preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Normalize (T, C, H, W) frame tensor to ViCLIP input format."""
        import torch.nn.functional as F

        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        elif frames.max() > 1.0:
            frames = frames.float() / 255.0
        else:
            frames = frames.float()

        if frames.shape[-2] != _VICLIP_SIZE or frames.shape[-1] != _VICLIP_SIZE:
            frames = F.interpolate(
                frames,
                size=(_VICLIP_SIZE, _VICLIP_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        mean = torch.tensor(_VICLIP_MEAN, device=frames.device).view(1, 3, 1, 1)
        std = torch.tensor(_VICLIP_STD, device=frames.device).view(1, 3, 1, 1)
        return (frames - mean) / std

    @torch.no_grad()
    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        all_embeddings = []

        for batch in tqdm(texts, disable=not show_progress_bar, desc="Text Encoding"):
            text_list = batch["text"]
            # get_text_features may process one string at a time (caching dict API)
            # so we call per-string and stack to handle both list and single-string APIs
            batch_feats = []
            for text in text_list:
                feat = self.model.get_text_features(text, self.tokenizer)
                if not isinstance(feat, torch.Tensor):
                    feat = (
                        feat.pooler_output
                        if hasattr(feat, "pooler_output")
                        else next(iter(feat.values()))
                    )
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                batch_feats.append(feat)
            features = torch.cat(batch_feats, dim=0)
            all_embeddings.append(features.cpu())

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def get_video_embeddings(
        self,
        videos: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        all_embeddings = []

        for batch in tqdm(videos, disable=not show_progress_bar, desc="Video Encoding"):
            processed = []
            for v in batch["video"]:
                if isinstance(v, torch.Tensor):
                    v = self._preprocess_frames(v)
                processed.append(v)

            # Stack to (B, T, C, H, W) and move to device
            video_tensor = torch.stack(processed, dim=0).to(self.device)
            features = self.model.get_vid_features(video_tensor)
            if not isinstance(features, torch.Tensor):
                features = (
                    features.pooler_output
                    if hasattr(features, "pooler_output")
                    else next(iter(features.values()))
                )
            all_embeddings.append(features.cpu())

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
                raise ValueError("Number of texts and videos must match")
            return text_embeddings + video_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif video_embeddings is not None:
            return video_embeddings

        raise ValueError(
            f"No supported modality found in dataset features: "
            f"{list(inputs.dataset.features.keys())}"
        )


_VICLIP_COMMON = dict(
    loader=ViCLIPWrapper,
    loader_kwargs=dict(num_frames=8),
    model_type=["dense"],
    languages=["eng-Latn"],
    release_date="2024-09-17",
    modalities=["video", "text"],
    license="cc-by-nc-sa-4.0",
    open_weights=True,
    public_training_code="https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid",
    public_training_data="https://huggingface.co/datasets/OpenGVLab/InternVid",
    framework=["PyTorch", "Transformers"],
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(),
    citation=VICLIP_CITATION,
)

viclip_large_patch14 = ModelMeta(
    name="OpenGVLab/ViCLIP-L-14-hf",
    revision="1652361522e1cb41c28cdfae870f690d00e7456b",
    n_parameters=427_616_513,
    n_embedding_parameters=2_162_688,
    memory_usage_mb=1632,
    max_tokens=77,
    embed_dim=768,
    reference="https://huggingface.co/OpenGVLab/ViCLIP-L-14-hf",
    extra_requirements_groups=["image", "video"],
    **_VICLIP_COMMON,
)

viclip_base_patch16 = ModelMeta(
    name="OpenGVLab/ViCLIP-B-16-hf",
    revision="8484a9cb5b1b86e43c3ded53abe7485f52d8b789",
    n_parameters=149_620_993,
    n_embedding_parameters=1_048_576,
    memory_usage_mb=571,
    max_tokens=77,
    embed_dim=512,
    reference="https://huggingface.co/OpenGVLab/ViCLIP-B-16-hf",
    extra_requirements_groups=["image", "video"],
    **_VICLIP_COMMON,
)

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


class XCLIPModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_frames: int = 8,
        **kwargs: Any,
    ):
        from transformers import XCLIPModel as HFXCLIPModel
        from transformers import XCLIPProcessor

        self.model_name = model_name
        self.device = device
        self.num_frames = num_frames
        self.model = HFXCLIPModel.from_pretrained(model_name, revision=revision).to(
            self.device
        )
        self.model.eval()
        self.processor = XCLIPProcessor.from_pretrained(model_name, revision=revision)

    @torch.no_grad()
    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        all_embeddings = []

        for batch in tqdm(texts, disable=not show_progress_bar, desc="Text Encoding"):
            inputs = self.processor(
                text=batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.model.get_text_features(**inputs)
            if hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output
            all_embeddings.append(text_features.cpu())

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
            # Collator returns [N, C, H, W] tensors; XCLIPProcessor expects
            # each video as a list of [H, W, C] numpy frames
            video_list = []
            for v in batch["video"]:
                frames = (
                    v.permute(0, 2, 3, 1).numpy() if isinstance(v, torch.Tensor) else v
                )
                video_list.append(list(frames))
            inputs = self.processor(
                videos=video_list,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            video_features = self.model.get_video_features(**inputs)
            if hasattr(video_features, "pooler_output"):
                video_features = video_features.pooler_output
            all_embeddings.append(video_features.cpu())

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
        elif text_embeddings is not None:
            return text_embeddings
        elif video_embeddings is not None:
            return video_embeddings

        raise ValueError(
            f"No supported modality found in dataset features: {list(inputs.dataset.features.keys())}"
        )


XCLIP_CITATION = """
@article{ni2022expanding,
  title={Expanding Language-Image Pretrained Models for General Video Recognition},
  author={Ni, Bolin and Peng, Houwen and Chen, Minghao and Zhang, Songyang and Meng, Gaofeng and Fu, Jianlong and Xiang, Shiming and Ling, Haibin},
  journal={arXiv preprint arXiv:2208.02816},
  year={2022}
}"""

_XCLIP_COMMON = dict(
    model_type=["dense"],
    languages=["eng-Latn"],
    release_date="2022-08-04",
    modalities=["video", "text"],
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers"],
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    citation=XCLIP_CITATION,
)

xclip_base_patch32 = ModelMeta(
    loader=XCLIPModel,
    name="microsoft/xclip-base-patch32",
    revision="a2e27a78a2b5d802e894b8a1ef14f3a8ce490963",
    n_parameters=196_585_855,
    n_embedding_parameters=1_048_576,
    memory_usage_mb=750,
    max_tokens=77,
    embed_dim=512,
    reference="https://huggingface.co/microsoft/xclip-base-patch32",
    loader_kwargs=dict(num_frames=8),
    **_XCLIP_COMMON,
)

xclip_base_patch16 = ModelMeta(
    loader=XCLIPModel,
    name="microsoft/xclip-base-patch16",
    revision="d6184e3fd8780d04c85d0f1eabe5f94bf44d98f6",
    n_parameters=194_929_426,
    n_embedding_parameters=1_048_576,
    memory_usage_mb=743,
    max_tokens=77,
    embed_dim=512,
    reference="https://huggingface.co/microsoft/xclip-base-patch16",
    loader_kwargs=dict(num_frames=8),
    **_XCLIP_COMMON,
)

xclip_large_patch14 = ModelMeta(
    loader=XCLIPModel,
    name="microsoft/xclip-large-patch14",
    revision="40f6d177e0a057a50ac69ac1de6b5938fd268601",
    n_parameters=575_673_934,
    n_embedding_parameters=2_162_688,
    memory_usage_mb=2196,
    max_tokens=77,
    embed_dim=768,
    reference="https://huggingface.co/microsoft/xclip-large-patch14",
    loader_kwargs=dict(num_frames=8),
    **_XCLIP_COMMON,
)

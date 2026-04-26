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


class VJEPA2Model(AbsEncoder):
    """MTEB wrapper for V-JEPA 2 video/image encoder models from Meta/FAIR.

    V-JEPA 2 models are vision-only encoders that produce dense embeddings
    for images and videos. Text encoding is not supported.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        num_frames: int = 64,
        device: str | None = None,
        **kwargs: Any,
    ):
        from transformers import AutoModel, AutoVideoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_frames = num_frames
        self.model = AutoModel.from_pretrained(model_name, revision=revision).to(
            self.device
        )
        self.model.eval()
        self.processor = AutoVideoProcessor.from_pretrained(
            model_name, revision=revision
        )

    @staticmethod
    def get_text_embeddings(
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        raise ValueError("V-JEPA 2 models only support image and video encoding.")

    @torch.inference_mode()
    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        import numpy as np

        all_embeddings = []
        for batch in tqdm(images, disable=not show_progress_bar, desc="Image Encoding"):
            # Each image is a separate single-frame "video" (may differ in size)
            videos = [[np.array(img)] for img in batch["image"]]
            inputs = self.processor(videos=videos, return_tensors="pt")
            pixel_values = inputs["pixel_values_videos"]
            # Shape: [batch, 1, C, H, W] -> repeat to [batch, num_frames, C, H, W]
            pixel_values = pixel_values.expand(-1, self.num_frames, -1, -1, -1)
            pixel_values = pixel_values.to(self.device)
            features = self.model.get_vision_features(pixel_values)
            all_embeddings.append(features.cpu())
        return torch.cat(all_embeddings, dim=0)

    @torch.inference_mode()
    def get_video_embeddings(
        self,
        videos: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        all_embeddings = []
        for batch in tqdm(videos, disable=not show_progress_bar, desc="Video Encoding"):
            inputs = self.processor(videos=batch["video"], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_vision_features(**inputs)
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
        features = inputs.dataset.features
        has_video = "video" in features
        has_image = "image" in features
        has_text = "text" in features

        if has_video:
            inputs.collate_fn = FramesCollator(num_frames=self.num_frames)
            return self.get_video_embeddings(inputs, **kwargs)
        elif has_image:
            return self.get_image_embeddings(inputs, **kwargs)
        elif has_text:
            return self.get_text_embeddings(inputs, **kwargs)
        raise ValueError("No image or video data found.")


_VJEPA2_CITATION = r"""@article{assran2025vjepa2,
  title={{V-JEPA 2}: Self-Supervised Video Models Enable Understanding, Generation, and Planning},
  author={Assran, Mahmoud and Bardes, Adrien and Castrejon, Lluis and Duval, Quentin
    and Garrido, Quentin and Hrinchuk, Oleksii and Koppula, Hema and LeCun, Yann
    and Liao, Yuxin and Mialon, Gr{\'e}goire and Misra, Ishan and Rabbat, Michael
    and Rizvi, Syed Talal and Sun, Hu and Tong, Shengbang and Touvron, Hugo
    and Vondrick, Carl and Xu, Xinlei},
  year={2025},
}"""

_VJEPA2_TRAINING_DATASETS: set[str] = set()

_VJEPA2_COMMON = dict(
    modalities=["image", "video"],
    model_type=["dense"],
    languages=["eng-Latn"],
    license="mit",
    open_weights=True,
    release_date="2025-06-10",
    framework=["PyTorch", "Transformers", "safetensors"],
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    n_embedding_parameters=0,
    max_tokens=None,
    public_training_code="https://github.com/facebookresearch/jepa",
    public_training_data=None,
    training_datasets=_VJEPA2_TRAINING_DATASETS,
    citation=_VJEPA2_CITATION,
)


vjepa2_vitl_fpc64_256 = ModelMeta(
    loader=VJEPA2Model,
    name="facebook/vjepa2-vitl-fpc64-256",
    revision="b3c1679b7c34d3255ef3547f27c7b226aefab26f",
    n_parameters=325_971_328,
    memory_usage_mb=1244,
    embed_dim=1024,
    reference="https://huggingface.co/facebook/vjepa2-vitl-fpc64-256",
    loader_kwargs={"num_frames": 64},
    **_VJEPA2_COMMON,
)

vjepa2_vith_fpc64_256 = ModelMeta(
    loader=VJEPA2Model,
    name="facebook/vjepa2-vith-fpc64-256",
    revision="b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5",
    n_parameters=653_930_880,
    memory_usage_mb=2495,
    embed_dim=1280,
    reference="https://huggingface.co/facebook/vjepa2-vith-fpc64-256",
    loader_kwargs={"num_frames": 64},
    **_VJEPA2_COMMON,
)

vjepa2_vitg_fpc64_256 = ModelMeta(
    loader=VJEPA2Model,
    name="facebook/vjepa2-vitg-fpc64-256",
    revision="875c192b7b704b87d1e1d99345769632dd5f739a",
    n_parameters=1_034_555_264,
    memory_usage_mb=3947,
    embed_dim=1408,
    reference="https://huggingface.co/facebook/vjepa2-vitg-fpc64-256",
    loader_kwargs={"num_frames": 64},
    **_VJEPA2_COMMON,
)

vjepa2_vitg_fpc64_384 = ModelMeta(
    loader=VJEPA2Model,
    name="facebook/vjepa2-vitg-fpc64-384",
    revision="12ca91694b230e0d4b5b0078af6f4ae1d51e933d",
    n_parameters=1_034_555_264,
    memory_usage_mb=3947,
    embed_dim=1408,
    reference="https://huggingface.co/facebook/vjepa2-vitg-fpc64-384",
    loader_kwargs={"num_frames": 64},
    **_VJEPA2_COMMON,
)

vjepa2_vitg_fpc64_384_ssv2 = ModelMeta(
    loader=VJEPA2Model,
    name="facebook/vjepa2-vitg-fpc64-384-ssv2",
    revision="9f5fd615cb6f79065a28edcf1cc3ef25010dddfa",
    n_parameters=1_128_049_454,
    memory_usage_mb=4304,
    embed_dim=1408,
    reference="https://huggingface.co/facebook/vjepa2-vitg-fpc64-384-ssv2",
    loader_kwargs={"num_frames": 64},
    **_VJEPA2_COMMON,
)

vjepa2_vitl_fpc16_256_ssv2 = ModelMeta(
    loader=VJEPA2Model,
    name="facebook/vjepa2-vitl-fpc16-256-ssv2",
    revision="4aa02df83918538fc21cfaf576382fa20e489a80",
    n_parameters=375_485_998,
    memory_usage_mb=1433,
    embed_dim=1024,
    reference="https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2",
    loader_kwargs={"num_frames": 16},
    **_VJEPA2_COMMON,
)

vjepa2_vitg_fpc32_384_diving48 = ModelMeta(
    loader=VJEPA2Model,
    name="facebook/vjepa2-vitg-fpc32-384-diving48",
    revision="0b48243375319bd8e03e3cd5560d957095429189",
    n_parameters=1_127_871_920,
    memory_usage_mb=4303,
    embed_dim=1408,
    reference="https://huggingface.co/facebook/vjepa2-vitg-fpc32-384-diving48",
    loader_kwargs={"num_frames": 32},
    **_VJEPA2_COMMON,
)

vjepa2_vitl_fpc32_256_diving48 = ModelMeta(
    loader=VJEPA2Model,
    name="facebook/vjepa2-vitl-fpc32-256-diving48",
    revision="71ae2a8b1ff5a297aeeaae9b5e64c7a2e5e6a633",
    n_parameters=375_356_848,
    memory_usage_mb=1432,
    embed_dim=1024,
    reference="https://huggingface.co/facebook/vjepa2-vitl-fpc32-256-diving48",
    loader_kwargs={"num_frames": 32},
    **_VJEPA2_COMMON,
)

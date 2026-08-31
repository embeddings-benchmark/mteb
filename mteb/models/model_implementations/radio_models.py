from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


RADIO_CITATION = """@InProceedings{Ranzinger_2024_CVPR,
    author    = {Ranzinger, Mike and Heinrich, Greg and Kautz, Jan and Molchanov, Pavlo},
    title     = {AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {12490-12500}
}

"""

# AM-RADIO distils CLIP, DINOv2 and SAM into a single backbone. The exact
# teacher datasets are not enumerated by NVIDIA, so this is left empty rather
# than guessed at.
RADIO_TRAINING_DATASETS: set[str] = set()


class RADIOModel(AbsEncoder):
    """Wrapper for NVIDIA's AM-RADIO vision foundation models.

    RADIO is vision-only: it distils several teachers (CLIP, DINOv2, SAM) into
    one backbone and has no text tower, so only image encoding is supported.

    The forward pass returns a `(summary, spatial_features)` tuple. `summary`
    is the model's own global image representation - analogous to a ViT CLS
    token, shape (B, C) - and is what NVIDIA documents for image-level tasks,
    so it is used directly rather than pooling the spatial features.

    Reference: https://github.com/NVlabs/RADIO
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        image_resolution: int | tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoModel

        self.device = device
        self.model = (
            AutoModel.from_pretrained(
                model_name, revision=revision, trust_remote_code=True
            )
            .eval()
            .to(self.device)
        )
        if image_resolution is None:
            height, width = self.model.preferred_resolution
        elif isinstance(image_resolution, int):
            height = width = image_resolution
        else:
            height, width = image_resolution
        resolution = self.model.get_nearest_supported_resolution(
            height=height, width=width
        )
        self.resolution = (int(resolution.height), int(resolution.width))

    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        import torch.nn.functional as F
        from torchvision.transforms.functional import pil_to_tensor

        all_image_embeddings = []

        with torch.inference_mode():
            for batch in tqdm(
                images, disable=not show_progress_bar, desc="Image Encoding"
            ):
                resized = []
                for image in batch["image"]:
                    if isinstance(image, torch.Tensor):
                        pixels = image
                    else:
                        pixels = pil_to_tensor(image.convert("RGB"))
                    # RADIO's input conditioner is RGB-only: drop any alpha
                    # channel and expand greyscale before normalisation.
                    if pixels.shape[0] == 1:
                        pixels = pixels.repeat(3, 1, 1)
                    elif pixels.shape[0] > 3:
                        pixels = pixels[:3]
                    pixels = pixels.to(self.device)
                    if pixels.dtype == torch.uint8:
                        pixels = pixels.float().div_(255.0)
                    else:
                        pixels = pixels.float()
                    resized.append(
                        F.interpolate(
                            pixels.unsqueeze(0),
                            size=self.resolution,
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
                pixel_values = torch.cat(resized)
                summary, _spatial_features = self.model(pixel_values)
                all_image_embeddings.append(summary.float().cpu())

        return torch.cat(all_image_embeddings, dim=0)

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
        return self.get_image_embeddings(inputs, **kwargs)


radio_b = ModelMeta(
    loader=RADIOModel,
    name="nvidia/RADIO-B",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="8a88fad34f4d9d397b91f546772daf1ea6edbc00",
    release_date="2024-07-23",
    modalities=["image"],
    n_parameters=98_233_353,
    n_embedding_parameters=0,
    memory_usage_mb=375,
    max_tokens=None,
    embed_dim=2304,
    license="https://github.com/NVlabs/RADIO/blob/main/LICENSE",  # NSCLv1
    open_weights=True,
    public_training_code="https://github.com/NVlabs/RADIO",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/nvidia/RADIO-B",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=RADIO_TRAINING_DATASETS,
    citation=RADIO_CITATION,
    extra_requirements_groups=["radio"],
)

radio_l = ModelMeta(
    loader=RADIOModel,
    name="nvidia/RADIO-L",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="e337932e3abb0c9d9f7d4bbcfad15aca783e91bc",
    release_date="2024-07-23",
    modalities=["image"],
    n_parameters=319_881_225,
    n_embedding_parameters=0,
    memory_usage_mb=1220,
    max_tokens=None,
    embed_dim=3072,
    license="https://github.com/NVlabs/RADIO/blob/main/LICENSE",  # NSCLv1
    open_weights=True,
    public_training_code="https://github.com/NVlabs/RADIO",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/nvidia/RADIO-L",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=RADIO_TRAINING_DATASETS,
    citation=RADIO_CITATION,
    extra_requirements_groups=["radio"],
)

radio_h = ModelMeta(
    loader=RADIOModel,
    name="nvidia/RADIO-H",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="d7fb8f9ae5ec05a7f8c639c1d83325c3b8cea37d",
    release_date="2024-10-18",
    modalities=["image"],
    n_parameters=651_642_889,
    n_embedding_parameters=0,
    memory_usage_mb=2486,
    max_tokens=None,
    embed_dim=3840,
    license="https://github.com/NVlabs/RADIO/blob/main/LICENSE",  # NSCLv1
    open_weights=True,
    public_training_code="https://github.com/NVlabs/RADIO",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/nvidia/RADIO-H",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=RADIO_TRAINING_DATASETS,
    citation=RADIO_CITATION,
    extra_requirements_groups=["radio"],
)

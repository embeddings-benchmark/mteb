from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn.functional import normalize
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType

TIPSV2_CITATION = """@article{cao2026tipsv2,
  title={TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment},
  author={Cao, Bingyi and Chen, Koert and Maninis, Kevis-Kokitsi and Chen, Kaifeng and Karpur, Arjun and Xia, Ye and Dua, Sahil and Dabral, Tanmaya and Han, Guangxing and Han, Bohyung and Ainslie, Joshua and Bewley, Alex and Jacob, Mithun and Wagner, René and Ramos, Washington and Choromanski, Krzysztof and Seyedhosseini, Mojtaba and Zhou, Howard and Araujo, André},
  journal={arXiv preprint arXiv:2604.12012},
  year={2026}
}"""

# TIPSv2 is trained on a filtered subset of WebLI, which is not an mteb task.
TIPSV2_TRAINING_DATASETS = {"WebLI"}


class TIPSv2Model(AbsEncoder):
    """Wrapper for Google DeepMind's TIPSv2 image-text encoders.

    TIPSv2 has both an image tower and a text tower, so it supports image, text
    and fused image+text encoding.

    Two model-specific details are handled here:

    1. Images are passed as floats in `[0, 1]` at 448x448 with no ImageNet
       normalization, matching the reference usage on the model card.
    2. `encode_image` returns an object whose `cls_token` has shape
       `(batch, 1, dim)`, so index 0 is taken to obtain the global embedding.
       This lives in the same space as `encode_text`, which returns
       `(batch, dim)` directly.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        from torchvision import transforms
        from transformers import AutoModel

        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),  # converts PIL [0,255] → tensor [0,1], no ImageNet normalization
            ]
        )

    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        all_text_embeddings = []
        with torch.no_grad():
            for batch in tqdm(
                texts, disable=not show_progress_bar, desc="Text Encoding"
            ):
                emb = self.model.encode_text(batch["text"])
                all_text_embeddings.append(normalize(emb, dim=-1).cpu())
        return torch.cat(all_text_embeddings, dim=0)

    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        all_image_embeddings = []
        with torch.no_grad():
            for batch in tqdm(
                images, disable=not show_progress_bar, desc="Image Encoding"
            ):
                pixel_values = torch.stack(
                    [self.transform(img.convert("RGB")) for img in batch["image"]]
                ).to(self.device)
                out = self.model.encode_image(pixel_values)
                # cls_token shape: (batch, 1, dim) — take index 0 for global embedding
                emb = normalize(out.cls_token[:, 0, :], dim=-1)
                all_image_embeddings.append(emb.cpu())
        return torch.cat(all_image_embeddings, dim=0)

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
        text_embeddings = None
        image_embeddings = None
        if "text" in inputs.dataset.features:
            text_embeddings = self.get_text_embeddings(inputs, **kwargs)
        if "image" in inputs.dataset.features:
            image_embeddings = self.get_image_embeddings(inputs, **kwargs)

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError("The number of texts and images must be equal")
            return text_embeddings + image_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings
        raise ValueError("No text or image features found in input")


tipsv2_b14 = ModelMeta(
    loader=TIPSv2Model,
    name="google/tipsv2-b14",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="23cfc936f39a74b5ae272149f703e3941b190529",
    release_date="2026-04-09",
    modalities=["image", "text"],
    n_parameters=195_948_288,
    n_embedding_parameters=24_576_000,
    memory_usage_mb=747,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=64,
    reference="https://huggingface.co/google/tipsv2-b14",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=False,
    adapted_from="google/tipsv2-g14",
    superseded_by=None,
    training_datasets=TIPSV2_TRAINING_DATASETS,
    public_training_code="https://github.com/google-deepmind/tips",
    public_training_data=None,
    citation=TIPSV2_CITATION,
    extra_requirements_groups=["tipsv2"],
)

tipsv2_l14 = ModelMeta(
    loader=TIPSv2Model,
    name="google/tipsv2-l14",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="3001ef12a69340be4a946cc4e2799ac5eb697818",
    release_date="2026-04-09",
    modalities=["image", "text"],
    n_parameters=487_941_120,
    n_embedding_parameters=32_768_000,
    memory_usage_mb=1861,
    embed_dim=1024,
    license="apache-2.0",
    max_tokens=64,
    reference="https://huggingface.co/google/tipsv2-l14",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=False,
    adapted_from="google/tipsv2-g14",
    superseded_by=None,
    training_datasets=TIPSV2_TRAINING_DATASETS,
    public_training_code="https://github.com/google-deepmind/tips",
    public_training_data=None,
    citation=TIPSV2_CITATION,
    extra_requirements_groups=["tipsv2"],
)

tipsv2_so400m14 = ModelMeta(
    loader=TIPSv2Model,
    name="google/tipsv2-so400m14",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="582b2c6814afea7eb6a64ccc2ce3a4b5af1036a0",
    release_date="2026-04-09",
    modalities=["image", "text"],
    n_parameters=861_726_816,
    n_embedding_parameters=36_864_000,
    memory_usage_mb=3287,
    embed_dim=1152,
    license="apache-2.0",
    max_tokens=64,
    reference="https://huggingface.co/google/tipsv2-so400m14",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=False,
    adapted_from="google/tipsv2-g14",
    superseded_by=None,
    training_datasets=TIPSV2_TRAINING_DATASETS,
    public_training_code="https://github.com/google-deepmind/tips",
    public_training_data=None,
    citation=TIPSV2_CITATION,
    extra_requirements_groups=["tipsv2"],
)

tipsv2_g14 = ModelMeta(
    loader=TIPSv2Model,
    name="google/tipsv2-g14",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="9b6d9f098466347ec47856eb281c69641b50899d",
    release_date="2026-04-09",
    modalities=["image", "text"],
    n_parameters=1_525_085_696,
    n_embedding_parameters=49_152_000,
    memory_usage_mb=5818,
    embed_dim=1536,
    license="apache-2.0",
    max_tokens=64,
    reference="https://huggingface.co/google/tipsv2-g14",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    training_datasets=TIPSV2_TRAINING_DATASETS,
    public_training_code="https://github.com/google-deepmind/tips",
    public_training_data=None,
    citation=TIPSV2_CITATION,
    extra_requirements_groups=["tipsv2"],
)

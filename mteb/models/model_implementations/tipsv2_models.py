from __future__ import annotations

from typing import TYPE_CHECKING, Any, Unpack

import torch
from torch.nn.functional import normalize
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType

TIPSV2_CITATION = """@article{berton2025tipsv2,
  title={TIPSv2: Unified, Scalable and Fast Vision-Language Encoders for Dense and Global Representations},
  author={Berton, Gabriele and Zhai, Xiaohua and Noci, Lorenzo and Grill, Jean-Bastien and Caron, Mathilde},
  journal={arXiv preprint arXiv:2604.12012},
  year={2025}
}"""


class TIPSv2Model(AbsEncoder):
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
                    [self.transform(img) for img in batch["image"]]
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
    languages=None,
    open_weights=True,
    revision="245de45054528d86029a06375bd7ba12a93f5b20",
    release_date="2025-04-16",
    modalities=["image", "text"],
    n_parameters=195_900_000,
    n_embedding_parameters=24_576_000,
    memory_usage_mb=747,
    embed_dim=768,
    license="cc-by-4.0",
    max_tokens=64,
    reference="https://huggingface.co/google/tipsv2-b14",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    training_datasets={"WebLI"},
    public_training_code="https://github.com/google-deepmind/tips",
    public_training_data=None,
    citation=TIPSV2_CITATION,
)

tipsv2_l14 = ModelMeta(
    loader=TIPSv2Model,
    name="google/tipsv2-l14",
    model_type=["dense"],
    languages=None,
    open_weights=True,
    revision="e1ff7bc8049120b87b9bec35d13b46177bfcac0d",
    release_date="2025-04-16",
    modalities=["image", "text"],
    n_parameters=487_900_000,
    n_embedding_parameters=32_768_000,
    memory_usage_mb=1861,
    embed_dim=1024,
    license="cc-by-4.0",
    max_tokens=64,
    reference="https://huggingface.co/google/tipsv2-l14",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=False,
    adapted_from="google/tipsv2-g14",
    superseded_by=None,
    training_datasets={"WebLI"},
    public_training_code="https://github.com/google-deepmind/tips",
    public_training_data=None,
    citation=TIPSV2_CITATION,
)

tipsv2_so400m14 = ModelMeta(
    loader=TIPSv2Model,
    name="google/tipsv2-so400m14",
    model_type=["dense"],
    languages=None,
    open_weights=True,
    revision="44ff952dcbdd0e5bddfc92a6fc7bf9313bae45df",
    release_date="2025-04-16",
    modalities=["image", "text"],
    n_parameters=861_700_000,
    n_embedding_parameters=36_864_000,
    memory_usage_mb=3287,
    embed_dim=1152,
    license="cc-by-4.0",
    max_tokens=64,
    reference="https://huggingface.co/google/tipsv2-so400m14",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=False,
    adapted_from="google/tipsv2-g14",
    superseded_by=None,
    training_datasets={"WebLI"},
    public_training_code="https://github.com/google-deepmind/tips",
    public_training_data=None,
    citation=TIPSV2_CITATION,
)

tipsv2_g14 = ModelMeta(
    loader=TIPSv2Model,
    name="google/tipsv2-g14",
    model_type=["dense"],
    languages=None,
    open_weights=True,
    revision="a4f58f8ccb1923e562ef64ea419501f7ae0a1438",
    release_date="2025-04-16",
    modalities=["image", "text"],
    n_parameters=1_500_000_000,
    n_embedding_parameters=49_152_000,
    memory_usage_mb=5818,
    embed_dim=1536,
    license="cc-by-4.0",
    max_tokens=64,
    reference="https://huggingface.co/google/tipsv2-g14",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    training_datasets={"WebLI"},
    public_training_code="https://github.com/google-deepmind/tips",
    public_training_data=None,
    citation=TIPSV2_CITATION,
)

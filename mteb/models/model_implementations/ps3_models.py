from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

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

PS3_CITATION = """@article{shi2025scaling,
  title={Scaling Vision Pre-Training to 4K Resolution},
  author={Shi, Baifeng and Li, Boyi and Cai, Han and Lu, Yao and Liu, Sifei and Pavone, Marco and Kautz, Jan and Han, Song and Darrell, Trevor and Molchanov, Pavlo and Yin, Hongxu},
  journal={arXiv preprint arXiv:2503.19903},
  year={2025}
}"""

# PS3 is pre-trained on 75M high-resolution images (38M natural images from DataComp and
# SA-1B, 37M document images from IDL and PDFA) with 282M automatically curated local
# region captions. None of these are mteb tasks.
PS3_TRAINING_DATASETS = {"DataComp", "SA-1B", "IDL", "PDFA"}


class PS3Model(AbsEncoder):
    """Wrapper for NVIDIA's PS3 (Scaling Vision Pre-Training to 4K Resolution) encoders.

    PS3 pairs a multi-scale vision tower with a CLIP-style text tower. The vision tower
    always encodes the image at low resolution first, then optionally selects and encodes
    a budget of high-resolution patches, either bottom-up (visual saliency) or top-down
    (conditioned on a text prompt embedding).

    Because mteb needs a single vector per input, only the pooled outputs are used.
    Two pooling modes are exposed through `pooling`:

    - `"global"` (default): the pooled low-resolution representation, matching the
      branch the pre-training loss uses for globally captioned samples.
    - `"high_res"`: the pooled representation over one round of bottom-up
      high-resolution patch selection (2560 patches).

    On ImageNetDog15Clustering the two score equivalently (NMI 0.600 vs 0.613 on a
    48-image probe), so `"global"` is the default on efficiency grounds: it skips the
    high-res forward pass entirely and is roughly an order of magnitude cheaper.

    Note that PS3 scores well below its SigLIP initialization on image clustering
    (see PR description), while zero-shot classification is unaffected
    (DTDZeroShot 0.61 for PS3-4K-SigLIP2). PS3 is pre-trained largely against local
    region captions rather than global image captions.

    Prompt-aware (top-down) selection is deliberately not used: it would make the image
    embedding depend on the query, which the single-vector encode interface cannot express.

    Both towers are L2-normalized so cosine similarity is comparable across modalities.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pooling: Literal["global", "high_res"] = "global",
        model_dtype: str = "float32",
        **kwargs: Any,
    ):
        from ps3 import PS3ImageProcessor, PS3TextModel, PS3Tokenizer, PS3VisionModel

        if pooling not in {"global", "high_res"}:
            raise ValueError(
                f"`pooling` must be either 'global' or 'high_res', got {pooling!r}."
            )

        self.model_name = model_name
        self.device = device
        self.pooling = pooling
        self.dtype = getattr(torch, model_dtype)

        self.vision_model = (
            PS3VisionModel.from_pretrained(model_name, revision=revision)
            .to(device=self.device, dtype=self.dtype)
            .eval()
        )
        self.text_model = (
            PS3TextModel.from_pretrained(model_name, revision=revision)
            .to(device=self.device, dtype=self.dtype)
            .eval()
        )
        self.processor = PS3ImageProcessor.from_pretrained(
            model_name, revision=revision
        )
        self.tokenizer = PS3Tokenizer.from_pretrained(model_name, revision=revision)
        # kept for parity with other wrappers that expose a single `.model`
        self.model = self.vision_model

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
                tokens = self.tokenizer(list(batch["text"])).to(self.device)
                emb = self.text_model(tokens).pooled_output
                all_text_embeddings.append(normalize(emb.float(), dim=-1).cpu())
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
                    [
                        self.processor(img.convert("RGB"))["pixel_values"][0]
                        for img in batch["image"]
                    ]
                ).to(device=self.device, dtype=self.dtype)
                if self.pooling == "global":
                    # pooled low-res features; identical to the branch the pre-training
                    # loss uses for globally captioned samples, but skips the high-res
                    # forward pass whose output would be discarded anyway
                    emb = self.vision_model.vision_model.forward_low_res(pixel_values)[
                        "x"
                    ]
                else:
                    # `output_hidden_states=False` returns the pooled embedding over one
                    # round of bottom-up high-res patch selection
                    emb = self.vision_model(
                        pixel_values, output_hidden_states=False
                    ).pooled_output
                all_image_embeddings.append(normalize(emb.float(), dim=-1).cpu())
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


ps3_1_5k_siglip = ModelMeta(
    loader=PS3Model,
    name="nvidia/PS3-1.5K-SigLIP",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="1f23460363a69645c9d5425838dc07a7413c1d6c",
    release_date="2025-05-20",
    modalities=["image", "text"],
    n_parameters=938312880,
    n_embedding_parameters=36864000,
    memory_usage_mb=3579,
    embed_dim=1152,
    license="https://huggingface.co/nvidia/PS3-1.5K-SigLIP/blob/main/LICENSE.md",
    max_tokens=64,
    reference="https://huggingface.co/nvidia/PS3-1.5K-SigLIP",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=False,
    adapted_from=None,
    superseded_by="nvidia/PS3-1.5K-SigLIP2",
    training_datasets=PS3_TRAINING_DATASETS,
    public_training_code="https://github.com/NVlabs/PS3",
    public_training_data=None,
    citation=PS3_CITATION,
    extra_requirements_groups=["ps3"],
)

ps3_4k_siglip = ModelMeta(
    loader=PS3Model,
    name="nvidia/PS3-4K-SigLIP",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="557d006054ae80d6b66340f2c1c0b8b3ffb4f49d",
    release_date="2025-05-20",
    modalities=["image", "text"],
    n_parameters=1028193584,
    n_embedding_parameters=36864000,
    memory_usage_mb=3922,
    embed_dim=1152,
    license="https://huggingface.co/nvidia/PS3-4K-SigLIP/blob/main/LICENSE.md",
    max_tokens=64,
    reference="https://huggingface.co/nvidia/PS3-4K-SigLIP",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=False,
    adapted_from="nvidia/PS3-1.5K-SigLIP",
    superseded_by="nvidia/PS3-4K-SigLIP2",
    training_datasets=PS3_TRAINING_DATASETS,
    public_training_code="https://github.com/NVlabs/PS3",
    public_training_data=None,
    citation=PS3_CITATION,
    extra_requirements_groups=["ps3"],
)

ps3_1_5k_siglip2 = ModelMeta(
    loader=PS3Model,
    name="nvidia/PS3-1.5K-SigLIP2",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="5a33ae8fd9dd011bfcb2eb04553bdf73c98e009c",
    release_date="2025-07-24",
    modalities=["image", "text"],
    n_parameters=1196360880,
    n_embedding_parameters=294912000,
    memory_usage_mb=4563,
    embed_dim=1152,
    license="https://huggingface.co/nvidia/PS3-1.5K-SigLIP2/blob/main/LICENSE.md",
    max_tokens=64,
    reference="https://huggingface.co/nvidia/PS3-1.5K-SigLIP2",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    training_datasets=PS3_TRAINING_DATASETS,
    public_training_code="https://github.com/NVlabs/PS3",
    public_training_data=None,
    citation=PS3_CITATION,
    extra_requirements_groups=["ps3"],
)

ps3_4k_siglip2 = ModelMeta(
    loader=PS3Model,
    name="nvidia/PS3-4K-SigLIP2",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="95dc088e70b32995289d51ac479eb3fce6d135e4",
    release_date="2025-07-24",
    modalities=["image", "text"],
    n_parameters=1286241584,
    n_embedding_parameters=294912000,
    memory_usage_mb=4906,
    embed_dim=1152,
    license="https://huggingface.co/nvidia/PS3-4K-SigLIP2/blob/main/LICENSE.md",
    max_tokens=64,
    reference="https://huggingface.co/nvidia/PS3-4K-SigLIP2",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=False,
    adapted_from="nvidia/PS3-1.5K-SigLIP2",
    superseded_by=None,
    training_datasets=PS3_TRAINING_DATASETS,
    public_training_code="https://github.com/NVlabs/PS3",
    public_training_data=None,
    citation=PS3_CITATION,
    extra_requirements_groups=["ps3"],
)

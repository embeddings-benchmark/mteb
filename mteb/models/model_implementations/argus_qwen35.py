from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class ArgusColQwen35Wrapper(AbsEncoder):
    """MTEB encoder for Argus-Colqwen3.5.

    Argus is a region-aware query-conditioned mixture-of-experts retriever
    built on the Qwen3.5-VL backbone. The model and processor are loaded via
    ``trust_remote_code=True`` from the released HF repo (the repo carries
    its own ``modeling_argus.py`` / ``processing_argus.py`` /
    ``configuration_argus.py``).

    The wrapper mirrors the in-tree ``OpsColQwen3Wrapper`` API so any task
    that already runs Ops-ColQwen3 will run Argus too. The only difference
    is that Argus's forward returns an ``ArgusOutput`` with an
    ``.embeddings`` attribute (rather than a bare tensor) — handled by
    :meth:`encode_input`.
    """

    def __init__(
        self,
        model_name: str = "DataScience-UIBK/Argus-Colqwen3.5-4b-v0",
        revision: str | None = None,
        device: str | None = None,
        attn_implementation: str | None = None,
        trust_remote_code: bool = True,
        max_num_visual_tokens: int = 2048,
        **kwargs: Any,
    ):
        from transformers.utils.import_utils import is_flash_attn_2_available

        if attn_implementation is None:
            attn_implementation = (
                "flash_attention_2" if is_flash_attn_2_available() else None
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.mdl = AutoModel.from_pretrained(
            model_name,
            device_map=self.device,
            attn_implementation=attn_implementation,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self.mdl.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            max_num_visual_tokens=max_num_visual_tokens,
        )

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
        text_embeddings = None
        image_embeddings = None

        if "text" in inputs.dataset.features:
            text_embeddings = self.get_text_embeddings(inputs, **kwargs)
        if "image" in inputs.dataset.features:
            image_embeddings = self.get_image_embeddings(inputs, **kwargs)

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            return text_embeddings + image_embeddings
        if text_embeddings is not None:
            return text_embeddings
        if image_embeddings is not None:
            return image_embeddings
        raise ValueError("No text or image inputs found")

    def encode_input(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.mdl(**inputs)
        # Argus returns ``ArgusOutput(embeddings=...)``; other ColQwen variants
        # return a plain tensor. Support both so this wrapper stays close to
        # ``OpsColQwen3Wrapper`` and can be re-used for the dense-baseline
        # variants we publish alongside Argus.
        emb = getattr(out, "embeddings", None)
        if emb is None:
            emb = out[0] if not isinstance(out, torch.Tensor) else out
        return emb

    def get_image_embeddings(
        self,
        images: DataLoader,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> torch.Tensor:
        import torchvision.transforms.functional as F
        from PIL import Image

        all_embeds: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(images, desc="Encoding images"):
                imgs = [
                    F.to_pil_image(b.to(self.device))
                    if not isinstance(b, Image.Image)
                    else b
                    for b in batch["image"]
                ]
                imgs = [img.convert("RGB") for img in imgs]
                inp = self.processor.process_images(imgs)
                inp = {k: v.to(self.device) for k, v in inp.items()}
                outs = self.encode_input(inp)
                all_embeds.extend(outs.cpu().to(torch.float32))

        return torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )

    def get_text_embeddings(
        self,
        texts: DataLoader,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> torch.Tensor:
        all_embeds: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(texts, desc="Encoding texts"):
                inp = self.processor.process_queries(batch["text"])
                inp = {k: v.to(self.device) for k, v in inp.items()}
                outs = self.encode_input(inp)
                all_embeds.extend(outs.cpu().to(torch.float32))

        return torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.processor.score_multi_vector(a, b, device=self.device)


ARGUS_TRAINING_DATA = {
    # ViDoRe train-set subsets used during distillation. Reported here so the
    # MTEB leaderboard correctly flags any test-set overlap.
    "VDRMultilingualRetrieval",
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
}

ARGUS_CITATION = """
@misc{argus2026,
  title  = {Argus: Region-Aware Query-Conditioned Mixture of Experts for Visual Document Retrieval},
  author = {DataScience-UIBK team},
  year   = {2026},
  url    = {https://huggingface.co/DataScience-UIBK/Argus-Colqwen3.5-4b-v0},
}"""


argus_colqwen35_4b = ModelMeta(
    loader=ArgusColQwen35Wrapper,
    name="DataScience-UIBK/Argus-Colqwen3.5-4b-v0",
    loader_kwargs=dict(
        max_num_visual_tokens=2048,
        trust_remote_code=True,
    ),
    languages=["eng-Latn"],
    revision="fedffec17bc28034ce77f3e99500c6864c4d4b6b",
    release_date="2026-04-29",
    modalities=["image", "text"],
    n_parameters=4_708_446_726,
    n_embedding_parameters=None,
    memory_usage_mb=8981,
    max_tokens=32768,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/DataScience-UIBK/Argus-Colqwen3.5-4b-v0",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=ARGUS_TRAINING_DATA,
    citation=ARGUS_CITATION,
    model_type=["late-interaction"],
)


argus_colqwen35_4b_bf16 = ModelMeta(
    loader=ArgusColQwen35Wrapper,
    name="DataScience-UIBK/Argus-Colqwen3.5-4b-v0-bf16",
    loader_kwargs=dict(
        max_num_visual_tokens=2048,
        trust_remote_code=True,
    ),
    languages=["eng-Latn"],
    revision="c88506c1bd05eb31ccf6a5c9b062bce0e8362520",
    release_date="2026-05-03",
    modalities=["image", "text"],
    n_parameters=4_708_446_726,
    n_embedding_parameters=None,
    memory_usage_mb=8981,
    max_tokens=32768,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/DataScience-UIBK/Argus-Colqwen3.5-4b-v0-bf16",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=ARGUS_TRAINING_DATA,
    citation=ARGUS_CITATION,
    model_type=["late-interaction"],
    adapted_from="DataScience-UIBK/Argus-Colqwen3.5-4b-v0",
)
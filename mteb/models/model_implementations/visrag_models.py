"""Implementation of the VisRAG-Ret visual document retriever by OpenBMB.

Paper: https://arxiv.org/abs/2410.10594
Model card: https://huggingface.co/openbmb/VisRAG-Ret

VisRAG-Ret is a single-vector dense retriever built on top of MiniCPM-V 2.0
(SigLIP-So400m vision tower + MiniCPM-2B language model). Both text queries
and image documents are projected into the same 2304-d space via weighted
mean pooling over the last hidden states, then L2-normalised so cosine and
dot-product scores coincide.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

# From the model card's reference usage:
# https://huggingface.co/openbmb/VisRAG-Ret#-usage
VISRAG_QUERY_INSTRUCTION = "Represent this query for retrieving relevant documents: "


def _weighted_mean_pooling(
    hidden: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Position-weighted mean pooling used by VisRAG-Ret.

    Later tokens get larger weight via the cumulative sum of the mask.
    Reproduces the snippet on the model card exactly.
    """
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    return s / d


class VisRAGRetWrapper(AbsEncoder):
    """Wrapper for OpenBMB's VisRAG-Ret visual document retriever."""

    def __init__(
        self,
        model_name: str = "openbmb/VisRAG-Ret",
        revision: str | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )

        config = AutoConfig.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )
        if (
            isinstance(getattr(config, "rope_scaling", None), dict)
            and "type" not in config.rope_scaling
        ):
            config.rope_scaling = None

        self.mdl = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **kwargs,
        ).to(self.device)
        self.mdl.eval()

    @torch.no_grad()
    def _encode(self, *, texts: list[str], images: list[Any]) -> torch.Tensor:
        """Run the custom VisRAG forward and apply the published pooling/norm."""
        outputs = self.mdl(text=texts, image=images, tokenizer=self.tokenizer)
        reps = _weighted_mean_pooling(outputs.last_hidden_state, outputs.attention_mask)
        return torch.nn.functional.normalize(reps, p=2, dim=1).to(torch.float32).cpu()

    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        all_embeds = []
        for batch in tqdm(texts, disable=not show_progress_bar, desc="Text Encoding"):
            queries = [VISRAG_QUERY_INSTRUCTION + t for t in batch["text"]]
            placeholders = [None] * len(queries)
            all_embeds.append(self._encode(texts=queries, images=placeholders))
        return torch.cat(all_embeds, dim=0)

    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        all_embeds = []
        for batch in tqdm(images, disable=not show_progress_bar, desc="Image Encoding"):
            imgs = [img.convert("RGB") for img in batch["image"]]
            placeholders = [""] * len(imgs)
            all_embeds.append(self._encode(texts=placeholders, images=imgs))
        return torch.cat(all_embeds, dim=0)

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
        raise ValueError("Inputs contain neither 'text' nor 'image' features.")


VISRAG_CITATION = """@misc{yu2024visrag,
  title         = {VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents},
  author        = {Shi Yu and Chaoyue Tang and Bokai Xu and Junbo Cui and Junhao Ran and Yukun Yan and Zhenghao Liu and Shuo Wang and Xu Han and Zhiyuan Liu and Maosong Sun},
  year          = {2024},
  eprint        = {2410.10594},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IR},
  url           = {https://arxiv.org/abs/2410.10594}
}"""

VISRAG_RET_TRAINING_DATASETS = {
    "VisRAGRetArxivQA",
    "VisRAGRetChartQA",
    "VisRAGRetMPDocVQA",
    "VisRAGRetInfoVQA",
    "VisRAGRetPlotQA",
    "VisRAGRetSlideVQA",
}

visrag_ret = ModelMeta(
    loader=VisRAGRetWrapper,
    loader_kwargs=dict(torch_dtype=torch.bfloat16),
    name="openbmb/VisRAG-Ret",
    revision="95ef596df871b606167cb7e4b7215caf1bfdf761",
    release_date="2024-10-14",
    languages=["eng-Latn"],
    n_parameters=3_434_965_792,
    n_embedding_parameters=284_488_704,
    memory_usage_mb=13_103,
    max_tokens=4096,
    embed_dim=2304,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/OpenBMB/VisRAG",
    public_training_data="https://huggingface.co/collections/openbmb/visrag-6717bbfb471bb018a49f1c69",
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/openbmb/VisRAG-Ret",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=VISRAG_RET_TRAINING_DATASETS,
    modalities=["image", "text"],
    model_type=["dense"],
    citation=VISRAG_CITATION,
    extra_requirements_groups=("visrag-ret",),
)

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

from mteb._requires_package import requires_image_dependencies
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class ColVec1Wrapper(AbsEncoder):
    """
    MTEB wrapper for ColVec1 (ColQwen3.5-based) retrieval models.

    Loads via AutoModel/AutoProcessor with trust_remote_code=True so no
    external library beyond transformers is required.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype | None = torch.bfloat16,
        **kwargs,
    ):
        requires_image_dependencies()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
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
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings
        raise ValueError("No text or image features found in inputs.")

    def _encode_inputs(self, encoded_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        vlm = getattr(self.model, "vlm", None)
        if vlm is not None:
            base = getattr(vlm, "model", vlm)
            if hasattr(base, "rope_deltas"):
                base.rope_deltas = None
        return self.model(**encoded_inputs)

    def get_image_embeddings(self, images, batch_size=32, show_progress_bar=True, **kwargs):
        import torchvision.transforms.functional as F
        from PIL import Image
        all_embeds = []
        with torch.no_grad():
            for batch in tqdm(images, disable=not show_progress_bar, desc="Encoding images"):
                imgs = [
                    F.to_pil_image(b.to(self.device)) if not isinstance(b, Image.Image) else b
                    for b in batch["image"]
                ]
                imgs = [img.convert("RGB") for img in imgs]
                inputs = self.processor.process_images(imgs)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self._encode_inputs(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))
        return torch.nn.utils.rnn.pad_sequence(all_embeds, batch_first=True, padding_value=0)

    def get_text_embeddings(self, texts, batch_size=32, show_progress_bar=True, **kwargs):
        all_embeds = []
        with torch.no_grad():
            for batch in tqdm(texts, disable=not show_progress_bar, desc="Encoding texts"):
                inputs = self.processor.process_queries(batch["text"])
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self._encode_inputs(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))
        return torch.nn.utils.rnn.pad_sequence(all_embeds, batch_first=True, padding_value=0)

    def similarity(self, a, b):
        a = [torch.as_tensor(x) for x in a]
        b = [torch.as_tensor(x) for x in b]
        return self.processor.score_multi_vector(a, b, device=self.device)


COLWEBAI_TRAINING_DATA = {
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
    "VisRAG-Ret-Train-Synthetic-data",
    "VisRAG-Ret-Train-In-domain-data",
    "VDRMultilingualRetrieval",
    "VidoreTabfquadRetrieval",
}

COLWEBAI_CITATION = """
@misc{webAI-ColVec1,
  title={webAI-ColVec1: Late-Interaction Multi-Vector Embedding Model for Visual Document Retrieval},
  author={webAI},
  year={2026},
  url={https://huggingface.co/webAI-Official/webAI-ColVec1-4b}
}
"""


colvec1_4b = ModelMeta(
    loader=ColVec1Wrapper,
    loader_kwargs=dict(torch_dtype=torch.bfloat16),
    name="webAI-Official/webAI-ColVec1-4b",
    revision='dce73882e6b89a01e702891a593f775dc5711929',
    release_date="2026-04-05",
    model_type=["late-interaction"],
    languages=["eng-Latn", "fra-Latn"],
    modalities=["image", "text"],
    n_parameters=4540904576,
    n_embedding_parameters=1639040,
    n_active_parameters_override=None,
    memory_usage_mb=8661,
    max_tokens=262144,
    embed_dim=640,
    license="multiple",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/webAI-Official/webAI-ColVec1-4b",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=False,
    training_datasets=COLWEBAI_TRAINING_DATA,
    adapted_from=None,
    superseded_by=None,
    citation=COLWEBAI_CITATION,
    contacts=None,
    output_dtypes=["bfloat16"],
    is_cross_encoder=False,
)


colvec1_9b = ModelMeta(
    loader=ColVec1Wrapper,
    loader_kwargs=dict(torch_dtype=torch.bfloat16),
    name="webAI-Official/webAI-ColVec1-9b",
    revision="3767539920b9132abb24cef2c88d42d81817e50b",
    release_date="2026-04-05",
    model_type=["late-interaction"],
    languages=["eng-Latn", "fra-Latn"],
    modalities=["image", "text"],
    n_parameters=9420302064,
    n_embedding_parameters=10488320,
    n_active_parameters_override=None,
    memory_usage_mb=17968,
    max_tokens=262144,
    embed_dim=2560,
    license="multiple",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/webAI-Official/webAI-ColVec1-9b",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=False,
    training_datasets=COLWEBAI_TRAINING_DATA,
    adapted_from=None,
    superseded_by=None,
    citation=COLWEBAI_CITATION,
    contacts=None,
    experiment_kwargs=None,
    output_dtypes=["bfloat16"],
    is_cross_encoder=False,
)
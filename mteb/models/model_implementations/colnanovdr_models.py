from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from tqdm.autonotebook import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

COLNANOVDR_CITATION = """@article{nanovdr2026,
  title={NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval},
  author={Liu, Zhuchenyang and Zhang, Yao and Xiao, Yu},
  journal={arXiv preprint arXiv:2603.12824},
  year={2026}
}"""

# Page resolution each teacher's own model card evaluates at, in visual tokens.
_MAX_VISUAL_TOKENS = 1792

# Train splits of these ViDoRe v1 sources are in the distillation data, through the
# ColPali training set (DocVQA, InfoVQA, TAT-DQA, arXivQA) and the VisRAG in-domain
# set (arXivQA, InfoVQA, MP-DocVQA, the latter built on DocVQA documents).
COLNANOVDR_TRAINING_DATA = {
    "VidoreArxivQARetrieval",
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
}


class ColNanoVDRWrapper(AbsEncoder):
    """Asymmetric late-interaction retrieval wrapper for ColNanoVDR.

    Queries go through a text-only multi-vector student; documents go through
    the frozen ColPali-style teacher the student was distilled into. Both sides
    emit a set of token vectors and are scored by MaxSim, so the student can be
    dropped into a corpus the teacher already indexed.

    The student folds its learned per-token weights into the vectors it emits,
    so they are deliberately not unit length and must not be renormalised.
    """

    def __init__(
        self,
        model_name: str,
        teacher_name: str,
        revision: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ):
        import torch

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.teacher_name = teacher_name

        from sentence_transformers import MultiVectorEncoder

        self.query_model = MultiVectorEncoder(
            model_name,
            revision=revision,
            trust_remote_code=True,
            device=self.device,
        )
        self.query_model.eval()

        # Document encoder: the frozen teacher, loaded on first use so that
        # query-only work never pays for it.
        self._doc_model = None
        self._doc_processor = None

    def _load_teacher(self) -> None:
        if self._doc_model is not None:
            return

        import torch
        from transformers import AutoModel, AutoProcessor

        self._doc_processor = AutoProcessor.from_pretrained(
            self.teacher_name,
            trust_remote_code=True,
            max_num_visual_tokens=_MAX_VISUAL_TOKENS,
        )
        self._doc_model = AutoModel.from_pretrained(
            self.teacher_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=self.device,
        ).eval()

    @staticmethod
    def _pad(blocks: list[torch.Tensor]) -> torch.Tensor:
        import torch

        return torch.nn.utils.rnn.pad_sequence(
            blocks, batch_first=True, padding_value=0.0
        )

    def _encode_queries(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
    ) -> Array:
        import torch

        texts = [text for batch in inputs for text in batch["text"]]
        embeddings = self.query_model.encode_query(
            texts,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=False,
        )
        return self._pad([torch.as_tensor(e).float().cpu() for e in embeddings])

    def _encode_documents(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
    ) -> Array:
        import torch
        import torchvision.transforms.functional as tv_functional
        from PIL import Image

        self._load_teacher()

        blocks: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(
                inputs, disable=not show_progress_bar, desc="Encoding docs"
            ):
                images = [
                    img
                    if isinstance(img, Image.Image)
                    else tv_functional.to_pil_image(img.cpu())
                    for img in batch["image"]
                ]
                images = [img.convert("RGB") for img in images]

                processed = self._doc_processor.process_images(images)
                processed = {
                    k: (v.to(self.device) if hasattr(v, "to") else v)
                    for k, v in processed.items()
                }
                out = self._doc_model(**processed)
                hidden = out if isinstance(out, torch.Tensor) else out[0]

                # The teacher returns raw hidden states, so the unit-sphere
                # normalisation MaxSim assumes has to happen here. Padding is
                # identified by the attention mask rather than by zero rows.
                mask = processed.get("attention_mask")
                for i in range(hidden.shape[0]):
                    vec = hidden[i]
                    if mask is not None:
                        vec = vec[mask[i].bool()]
                    vec = vec[vec.float().norm(dim=-1) > 1e-6]
                    vec = vec.float()
                    vec = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # noqa: PLR6104
                    blocks.append(vec.cpu())

        return self._pad(blocks)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        from mteb.types import PromptType

        if (
            prompt_type in (PromptType.query, None)  # noqa: PLR6201
            and "image" in inputs.dataset.features
        ):
            raise ValueError(
                f"ColNanoVDR only supports text queries, but task "
                f"'{task_metadata.name}' provides image inputs for queries. "
                f"ColNanoVDR is a text-query -> image-document retrieval model "
                f"and does not support image-query or image-classification tasks."
            )

        if prompt_type == PromptType.document:
            return self._encode_documents(inputs, show_progress_bar=show_progress_bar)
        return self._encode_queries(inputs, show_progress_bar=show_progress_bar)


colnanovdr_colvec_4b = ModelMeta(
    loader=ColNanoVDRWrapper,
    loader_kwargs=dict(teacher_name="webAI-Official/webAI-ColVec1.1-4b"),
    name="nanovdr/ColNanoVDR-Q-Ettin150M-ColVec4B-640-ML",
    model_type=["late-interaction"],
    languages=["eng-Latn", "deu-Latn", "fra-Latn", "spa-Latn", "ita-Latn", "por-Latn"],
    open_weights=True,
    revision="7c816c0ef9a351f085a2e4138716ea40c77721f5",
    release_date="2026-09-20",
    modalities=["text", "image"],
    n_parameters=150_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=601,
    embed_dim=640,
    license="https://huggingface.co/webAI-Official/webAI-ColVec1.1-4b/blob/main/LICENSE.md",
    max_tokens=512,
    reference="https://huggingface.co/nanovdr/ColNanoVDR-Q-Ettin150M-ColVec4B-640-ML",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/Ryenhails/NanoVDR",
    public_training_data="https://huggingface.co/datasets/nanovdr/NanoVDR-Train",
    training_datasets=COLNANOVDR_TRAINING_DATA,
    citation=COLNANOVDR_CITATION,
    extra_requirements_groups=["colnanovdr"],
)

colnanovdr_colvec_8b = ModelMeta(
    loader=ColNanoVDRWrapper,
    loader_kwargs=dict(teacher_name="webAI-Official/webAI-ColVec1.1-8b"),
    name="nanovdr/ColNanoVDR-Q-Ettin150M-ColVec8B-640-ML",
    model_type=["late-interaction"],
    languages=["eng-Latn", "deu-Latn", "fra-Latn", "spa-Latn", "ita-Latn", "por-Latn"],
    open_weights=True,
    revision="cc7671cb72d455d02a4885d3b0e9b2512a82cb4f",
    release_date="2026-09-20",
    modalities=["text", "image"],
    n_parameters=150_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=601,
    embed_dim=640,
    license="https://huggingface.co/webAI-Official/webAI-ColVec1.1-8b/blob/main/LICENSE.md",
    max_tokens=512,
    reference="https://huggingface.co/nanovdr/ColNanoVDR-Q-Ettin150M-ColVec8B-640-ML",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/Ryenhails/NanoVDR",
    public_training_data="https://huggingface.co/datasets/nanovdr/NanoVDR-Train",
    training_datasets=COLNANOVDR_TRAINING_DATA,
    citation=COLNANOVDR_CITATION,
    extra_requirements_groups=["colnanovdr"],
)

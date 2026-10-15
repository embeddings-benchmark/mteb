from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
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
    the frozen teacher the student was distilled into, loaded from the
    teacher's own MTEB entry so that its revision, processor settings and
    handling of text and image documents are exactly those of the teacher.
    Both sides emit a set of token vectors and are scored by MaxSim, so the
    student can be dropped into a corpus the teacher already indexed.

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
        from sentence_transformers import MultiVectorEncoder

        from mteb.models.get_model_meta import get_model_meta

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.query_model = MultiVectorEncoder(
            model_name,
            revision=revision,
            trust_remote_code=True,
            device=self.device,
        )
        self.query_model.eval()

        self.document_model = get_model_meta(teacher_name).load_model(
            device=self.device
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
        return torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(e).float().cpu() for e in embeddings],
            batch_first=True,
            padding_value=0.0,
        )

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

        if prompt_type == PromptType.document:
            return self.document_model.encode(
                inputs,
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=prompt_type,
                show_progress_bar=show_progress_bar,
                **kwargs,
            )

        if "image" in inputs.dataset.features:
            raise ValueError(
                f"ColNanoVDR only supports text queries, but task "
                f"'{task_metadata.name}' provides image inputs for queries."
            )
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
    n_parameters=149_506_561,
    n_embedding_parameters=38_682_624,
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
    extra_requirements_groups=["colnanovdr", "colvec1_1"],
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
    n_parameters=149_506_561,
    n_embedding_parameters=38_682_624,
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
    extra_requirements_groups=["colnanovdr", "colvec1_1"],
)

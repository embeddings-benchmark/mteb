from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import (
    MultiVectorWrapper,
    _concatenate_ragged_batches,
    _encode_batches,
    _select_encode_function,
)
from mteb.types import OutputDType

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType


def _to_half_cpu_in_place(batch: list[torch.Tensor]) -> list[torch.Tensor]:
    """Convert token vectors to FP16 and move them to the CPU one at a time, replacing them in `batch`.

    Text inputs are encoded as a whole corpus chunk in one call, so building a new list would keep
    both copies of the whole chunk alive at once. On the CPU they no longer occupy device memory;
    `MultiVectorWrapper.similarity` moves them back in blocks for scoring.
    """
    for index, vector in enumerate(batch):
        batch[index] = vector.half().cpu()
    return batch


class TopkEmbedWrapper(MultiVectorWrapper):
    """MultiVectorWrapper for the topk-embed-v1 models.

    The models have no image+text fusion, so an input with both columns (ViDoRe v3.1: page image
    and OCR markdown) is encoded from its text only. Token vectors are kept in FP16, which holds the
    BF16 model outputs exactly and keeps MaxSim's token similarities finer than BF16.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        image_token_budget: int = 2048,
        corpus_chunk_size: int = 1024,
        **kwargs: Any,
    ):
        super().__init__(
            model_name,
            revision=revision,
            corpus_chunk_size=corpus_chunk_size,
            trust_remote_code=True,
            config_kwargs={"image_token_budget": image_token_budget},
            **kwargs,
        )
        applied = self.model[0].config.image_token_budget
        if applied != image_token_budget:
            raise ValueError(
                f"image_token_budget={image_token_budget} was not applied; the model uses {applied}"
            )

    def _encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        modality = "text" if "text" in inputs.dataset.features else "image"
        return _encode_batches(
            inputs,
            is_multimodal=modality == "image",
            encode_function=_select_encode_function(self.model, prompt_type),
            prompt=None,
            modalities=[modality],
            postprocess_batch=_to_half_cpu_in_place,
            concatenate_batches=_concatenate_ragged_batches,
            encode_text_per_batch=True,
            **kwargs,
        )


TOPK_EMBED_V1_LANGUAGES = [
    "eng-Latn",
    "fra-Latn",
    "deu-Latn",
    "spa-Latn",
    "ita-Latn",
    "por-Latn",
]

TOPK_EMBED_V1_SOURCE_DATASETS = {
    "ArguAna",
    "DBPedia",
    "ESCIReranking",
    "FEVER",
    "FEVER-NL",
    "FiQA2018",
    "GerDaLIR",
    "HotpotQA",
    "HotpotQA-NL",
    "MultiLongDocRetrieval",
    "MIRACLRetrieval",
    "MKQARetrieval",
    "MMarcoRetrievalMultilingual",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "NarrativeQARetrieval",
    "QASPER",
    "SciFact",
    "VidoreArxivQARetrieval",
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "WebFAQRetrieval",
    # not in mteb
    # "2WikiMultihopQA"
    # "AgentIR"
    # "CaseHOLD"
    # "ChartQA"
    # "CLIRMatrix"
    # "CORD-19"
    # "Financial-QA-10K"
    # "gooaq_qa"
    # "MedicalQA_ru"
    # "MedMCQA"
    # "MedQA"
    # "MedQuAD"
    # "MQA"
    # "MuSiQue"
    # "PlotQA"
    # "PubMedQA"
    # "SlideVQA"
    # "SPECTER"
    # "squadv2"
    # "stackexchange_qa"
    # "trivia"
    # "WikiOmnia"
}

topk_embed_v1_xsmall = ModelMeta(
    loader=TopkEmbedWrapper,
    loader_kwargs={"image_token_budget": 2048},
    name="topk-io/topk-embed-v1-xsmall",
    revision="d09d8a7a8cdd6c287f792b3c4d7b41233d46e66a",
    release_date="2026-09-23",
    model_type=["late-interaction"],
    languages=TOPK_EMBED_V1_LANGUAGES,
    modalities=["image", "text"],
    n_parameters=854_034_496,
    n_embedding_parameters=254_279_680,
    memory_usage_mb=1629,
    max_tokens=8192,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers", "Transformers", "safetensors"],
    reference="https://huggingface.co/topk-io/topk-embed-v1-xsmall",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=False,
    training_datasets=TOPK_EMBED_V1_SOURCE_DATASETS,
    adapted_from="Qwen/Qwen3.5-0.8B",
    superseded_by=None,
    output_dtypes=OutputDType.FLOAT16,
    extra_requirements_groups=["topk-embed"],
)

topk_embed_v1_small = ModelMeta(
    loader=TopkEmbedWrapper,
    loader_kwargs={"image_token_budget": 2048},
    name="topk-io/topk-embed-v1-small",
    revision="33b15d544d74f29d04cdb97adeb9fd0e52da5fb7",
    release_date="2026-09-23",
    model_type=["late-interaction"],
    languages=TOPK_EMBED_V1_LANGUAGES,
    modalities=["image", "text"],
    n_parameters=2_217_435_968,
    n_embedding_parameters=508_559_360,
    memory_usage_mb=4229,
    max_tokens=8192,
    embed_dim=2048,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers", "Transformers", "safetensors"],
    reference="https://huggingface.co/topk-io/topk-embed-v1-small",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=False,
    training_datasets=TOPK_EMBED_V1_SOURCE_DATASETS,
    adapted_from="Qwen/Qwen3.5-2B",
    superseded_by=None,
    output_dtypes=OutputDType.FLOAT16,
    extra_requirements_groups=["topk-embed"],
)

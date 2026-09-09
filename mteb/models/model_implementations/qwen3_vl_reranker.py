from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING, Any

from mteb.models.model_implementations.qwen3_vl_embedding_models import (
    QWEN3_VL_EMBEDDING_CITATION,
)
from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import CrossEncoderWrapper

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

DEFAULT_INSTRUCTION = "Retrieve images or text relevant to the user's query."


QWEN3_VL_LANGUAGES = [
    "eng-Latn",
    "cmn-Hans",
    "spa-Latn",
    "fra-Latn",
    "ara-Arab",
    "por-Latn",
    "rus-Cyrl",
    "urd-Arab",
    "ind-Latn",
    "deu-Latn",
    "jpn-Jpan",
    "vie-Latn",
    "tur-Latn",
    "kor-Hang",
    "fas-Arab",
    "ita-Latn",
    "tha-Thai",
    "pol-Latn",
    "ukr-Cyrl",
    "uzb-Latn",
    "ron-Latn",
    "nld-Latn",
    "kaz-Cyrl",
    "ell-Grek",
    "ces-Latn",
    "swe-Latn",
    "srp-Cyrl",
    "heb-Hebr",
    "dan-Latn",
    "fin-Latn",
    "nor-Latn",
    "slv-Latn",
    "gle-Latn",
]


class Qwen3VLRerankerWrapper(CrossEncoderWrapper):
    """Wrapper for the Qwen3-VL-Reranker series.

    The reranker scores a (query, document) pair jointly, where either side may
    be text, an image, a video, or a mixture of these. It is exposed as a
    standard `sentence_transformers.CrossEncoder`, so the multimodal input
    collection in `CrossEncoderWrapper` is reused as-is; this subclass only
    supplies the task-level instruction. Pass `use_instructions=False` to
    score pairs without one.

    Reference implementation: https://github.com/QwenLM/Qwen3-VL-Embedding
    """

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        device: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        fps: float | None = 2.0,
        max_frames: int | None = 64,
        num_frames: int | None = None,
        use_instructions: bool = True,
        **kwargs: Any,
    ) -> None:
        # Default to the checkpoint's own preprocessor_config.json. Forcing a
        # pixel budget changes how images are resized and measurably shifts
        # scores away from the reference implementation, so it is opt-in only
        # (useful if you need to bound memory on a small GPU).
        processor_kwargs = dict(kwargs.pop("processor_kwargs", {}) or {})
        if min_pixels is not None:
            processor_kwargs.setdefault("min_pixels", min_pixels)
        if max_pixels is not None:
            processor_kwargs.setdefault("max_pixels", max_pixels)
        if processor_kwargs:
            kwargs["processor_kwargs"] = processor_kwargs

        super().__init__(
            model,
            revision=revision,
            device=device,
            fps=fps,
            max_frames=max_frames,
            num_frames=num_frames,
            **kwargs,
        )
        self.use_instructions = use_instructions

    @staticmethod
    def _normalize_instruction(instruction: str) -> str:
        instruction = instruction.strip()
        # Mirrors Qwen3VLEmbeddingWrapper: append a period if the instruction
        # does not already end in punctuation.
        if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
            instruction += "."
        return instruction

    def get_task_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None = None,
    ) -> str:
        """Resolve the single task-level instruction passed to the reranker.

        Unlike a bi-encoder, the reranker consumes the query and the document in
        one forward pass, so it takes one instruction describing the relevance
        criterion rather than a per-side prefix. When a task defines separate
        query/document prompts, the query-side prompt is used.
        """
        prompt = task_metadata.prompt
        if isinstance(prompt, dict):
            prompt = prompt.get("query") or next(iter(prompt.values()), None)
        if not prompt:
            return DEFAULT_INSTRUCTION
        return self._normalize_instruction(prompt)

    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        if self.use_instructions:
            kwargs.setdefault(
                "prompt", self.get_task_instruction(task_metadata, prompt_type)
            )
        return super().predict(
            inputs1,
            inputs2,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )


qwen3_vl_reranker_2b = ModelMeta(
    loader=Qwen3VLRerankerWrapper,
    name="Qwen/Qwen3-VL-Reranker-2B",
    model_type=["cross-encoder"],
    languages=QWEN3_VL_LANGUAGES,
    open_weights=True,
    revision="4bd860ac4f15ad1897a214615cccc700f8f71818",
    release_date="2026-01-07",
    modalities=["image", "text", "video"],
    n_parameters=2_127_532_032,
    n_embedding_parameters=311_164_928,
    memory_usage_mb=4058,
    embed_dim=None,  # not applicable: outputs a scalar relevance score
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B",
    similarity_fn_name=None,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=QWEN3_VL_EMBEDDING_CITATION,
    extra_requirements_groups=["multimodal-sbert"],
)

qwen3_vl_reranker_8b = ModelMeta(
    loader=Qwen3VLRerankerWrapper,
    name="Qwen/Qwen3-VL-Reranker-8B",
    model_type=["cross-encoder"],
    languages=QWEN3_VL_LANGUAGES,
    open_weights=True,
    revision="b212dc8c91a8164aef1ea2de9c1a867611e75c04",
    release_date="2026-01-07",
    modalities=["image", "text", "video"],
    n_parameters=8_767_123_696,
    n_embedding_parameters=622_329_856,
    memory_usage_mb=16722,
    embed_dim=None,  # not applicable: outputs a scalar relevance score
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B",
    similarity_fn_name=None,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=QWEN3_VL_EMBEDDING_CITATION,
    extra_requirements_groups=["multimodal-sbert"],
)

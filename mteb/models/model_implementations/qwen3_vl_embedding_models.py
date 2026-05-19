from __future__ import annotations

from typing import Any

from mteb.models.instruct_wrapper import MultimodalInstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta, ScoringFunction

QWEN3_VL_EMBEDDING_CITATION = """@article{qwen3vlembedding,
  title={Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking},
  author={Li, Mingxin and Zhang, Yanzhao and Long, Dingkun and Chen Keqin and Song, Sibo and Bai, Shuai and Yang, Zhibo and Xie, Pengjun and Yang, An and Liu, Dayiheng and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2601.04720},
  year={2026}
}"""

IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
DEFAULT_INSTRUCTION = "Represent the user's input."


class Qwen3VLEmbeddingWrapper(MultimodalInstructSentenceTransformerModel):
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        fps: float | None = 2.0,
        max_frames: int | None = 64,
        num_frames: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name,
            revision=revision,
            device=device,
            fps=fps,
            max_frames=max_frames,
            num_frames=num_frames,
            **kwargs,
        )
        self.model[0].processing_kwargs.update(
            {
                "image": {
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                "video": {
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "do_sample_frames": False,
                },
            }
        )


qwen3_vl_embedding_2b = ModelMeta(
    loader=Qwen3VLEmbeddingWrapper,
    loader_kwargs={
        "trust_remote_code": True,
    },
    name="Qwen/Qwen3-VL-Embedding-2B",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="2a50926d213628c727f38025982a76f655673f54",
    release_date="2026-01-08",
    modalities=["image", "text", "video"],
    n_parameters=2_127_532_032,
    n_embedding_parameters=311_164_928,
    memory_usage_mb=7629,
    embed_dim=2048,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=QWEN3_VL_EMBEDDING_CITATION,
    extra_requirements_groups=["multimodal-sbert"],
)

qwen3_vl_embedding_8b = ModelMeta(
    loader=Qwen3VLEmbeddingWrapper,
    loader_kwargs={
        "trust_remote_code": True,
    },
    name="Qwen/Qwen3-VL-Embedding-8B",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="418d03899aeb9d954e776dfa762006616519a914",
    release_date="2026-01-08",
    modalities=["image", "text", "video"],
    n_parameters=8_144_793_840,
    n_embedding_parameters=622_329_856,
    memory_usage_mb=30518,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=QWEN3_VL_EMBEDDING_CITATION,
    extra_requirements_groups=["multimodal-sbert"],
)

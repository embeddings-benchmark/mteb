from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import (
    SentenceTransformerEncoderWrapper,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class WeMMEncoderWrapper(SentenceTransformerEncoderWrapper):
    """Custom SentenceTransformer encoder wrapper for WeChat WeMM Embedding models."""

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
        import numpy as np
        from torchvision.transforms.functional import to_pil_image
        from tqdm.auto import tqdm

        features = inputs.dataset.features
        has_video = "video" in features
        is_multimodal = has_video or "image" in features

        if has_video:
            from mteb.models.modality_collators import VideoCollator

            inputs.collate_fn = VideoCollator(
                target_sampling_rate=self.target_sampling_rate or 16000,
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
            )

        instruction = self.get_task_instruction(task_metadata, prompt_type)
        all_embeddings = []

        # Process batches from MTEB's DataLoader
        for batch in tqdm(inputs, desc="Encoding batches"):
            texts = batch.get("text")
            images = batch.get("image")
            videos = batch.get("video")

            if videos is not None:
                videos = [
                    [
                        to_pil_image(
                            f.permute(2, 0, 1)
                            if f.ndim == 3 and f.shape[-1] == 3
                            else f
                        )
                        for f in v.cpu()
                    ]
                    for v in videos
                ]

            batch_size = len(texts or images or videos or [])
            batched_input = []

            for i in range(batch_size):
                text_content = " ".join(
                    [t for t in [instruction, texts[i] if texts else None] if t]
                )

                if is_multimodal:
                    sample = {}
                    if images and images[i] is not None:
                        sample["image"] = images[i]
                    if videos and videos[i] is not None:
                        sample["video"] = videos[i]
                    if text_content:
                        sample["text"] = text_content
                    batched_input.append(sample)
                else:
                    batched_input.append(text_content)

            embeddings = self.model.encode(batched_input, **kwargs)
            all_embeddings.append(embeddings)

        return cast("Array", np.concatenate(all_embeddings, axis=0))


# --- Private module-level constants ---

_WEMM_CITATION = """@article{wemm-embedding,
      title={WeMM-Embedding: WeChat Multi-Modal Embedding Technical Report},
      author={Junjie Zhou and Ke Mei and Lei Li and Tianyi Wang and Fengyun Rao and Jing Lyu},
      year={2026},
      eprint={2608.24053},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.24053},
}"""

_WEMM_LOADER_KWARGS = dict(
    trust_remote_code=True,
    fps=2.0,
    max_frames=64,
)


wemm_embedding_2b = ModelMeta(
    loader=WeMMEncoderWrapper,
    loader_kwargs=_WEMM_LOADER_KWARGS,
    name="tencent/WeMM-Embedding-2B",
    model_type=["dense"],
    modalities=["text", "image", "video"],
    languages=["zho-Hans", "eng-Latn"],
    open_weights=True,
    revision="df8094e5caf29083d9cac28e96fad6cfbe3ee57f",
    release_date="2026-08-25",
    n_parameters=2_210_000_000,
    n_embedding_parameters=508_063_744,
    memory_usage_mb=4215,
    max_tokens=262144,
    embed_dim=[64, 128, 256, 512, 1024, 2048],
    license="apache-2.0",
    reference="https://huggingface.co/tencent/WeMM-Embedding-2B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    citation=_WEMM_CITATION,
    public_training_code="https://github.com/Tencent/WeMM-Embedding",
    public_training_data=None,
    training_datasets=None,
    extra_requirements_groups=["wemm"],
)

wemm_embedding_4b = ModelMeta(
    loader=WeMMEncoderWrapper,
    loader_kwargs=_WEMM_LOADER_KWARGS,
    name="tencent/WeMM-Embedding-4B",
    model_type=["dense"],
    modalities=["text", "image", "video"],
    languages=["zho-Hans", "eng-Latn"],
    open_weights=True,
    revision="a28b25c5d18cf71ec46b115e06ea79ab00ee4819",
    release_date="2026-08-25",
    n_parameters=4_250_000_000,
    n_embedding_parameters=635_079_680,
    memory_usage_mb=8106,
    max_tokens=262144,
    embed_dim=[64, 128, 256, 512, 1024, 2560],
    license="apache-2.0",
    reference="https://huggingface.co/tencent/WeMM-Embedding-4B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    citation=_WEMM_CITATION,
    public_training_code="https://github.com/Tencent/WeMM-Embedding",
    public_training_data=None,
    training_datasets=None,
    extra_requirements_groups=["wemm"],
)

wemm_embedding_9b = ModelMeta(
    loader=WeMMEncoderWrapper,
    loader_kwargs=_WEMM_LOADER_KWARGS,
    name="tencent/WeMM-Embedding-9B",
    model_type=["dense"],
    modalities=["text", "image", "video"],
    languages=["zho-Hans", "eng-Latn"],
    open_weights=True,
    revision="59f673e3d53c4f71bb364adf344536dc1c2bc95e",
    release_date="2026-08-25",
    n_parameters=9_000_000_000,
    n_embedding_parameters=1_016_127_488,
    memory_usage_mb=17166,
    max_tokens=262144,
    embed_dim=[64, 128, 256, 512, 1024, 2048, 4096],
    license="apache-2.0",
    reference="https://huggingface.co/tencent/WeMM-Embedding-9B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    citation=_WEMM_CITATION,
    public_training_code="https://github.com/Tencent/WeMM-Embedding",
    public_training_data=None,
    training_datasets=None,
    extra_requirements_groups=["wemm"],
)

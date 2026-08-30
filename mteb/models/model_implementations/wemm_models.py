from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import (
    SentenceTransformerEncoderWrapper,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

WEMM_CITATION = """@article{wemm-embedding,
      title={WeMM-Embedding: WeChat Multi-Modal Embedding Technical Report},
      author={Junjie Zhou and Ke Mei and Lei Li and Tianyi Wang and Fengyun Rao and Jing Lyu},
      year={2026},
      eprint={2608.24053},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.24053},
}"""


class WeMMEncoderWrapper(SentenceTransformerEncoderWrapper):
    """Custom SentenceTransformer encoder wrapper for WeChat WeMM Embedding models."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        fps: float | None = 2.0,
        max_frames: int | None = 64,
        num_frames: int | None = None,
        target_sampling_rate: int = 16000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            revision=revision,
            device=device,
            **kwargs,
        )
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.target_sampling_rate = target_sampling_rate

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
        # A. Setup standard MTEB VideoCollator on the inputs if video is present
        features = inputs.dataset.features
        has_video = "video" in features
        is_multimodal = has_video or "image" in features

        if has_video:
            from mteb.models.modality_collators import VideoCollator

            inputs.collate_fn = VideoCollator(
                target_sampling_rate=self.target_sampling_rate,
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
            )

        instruction = self.get_task_instruction(task_metadata, prompt_type)
        all_embeddings = []

        # B. Process batches from MTEB's DataLoader
        for batch in tqdm(inputs, desc="Encoding batches"):
            texts = batch.get("text")
            images = batch.get("image")
            videos = batch.get("video")

            # Inline video frame tensor conversion (using compact list comprehension)
            if videos is not None:
                videos = [
                    [
                        # Convert HWC tensor format to CHW before calling to_pil_image
                        to_pil_image(
                            f.permute(2, 0, 1)
                            if f.ndim == 3 and f.shape[-1] == 3
                            else f
                        )
                        for f in v.cpu()
                    ]
                    if isinstance(v, torch.Tensor)
                    else v
                    for v in videos
                ]

            # C. Build unified list of inputs (string for text, dict for multimodal)
            batch_size = (
                len(texts) if texts else (len(images) if images else len(videos))
            )
            batched_input = []

            for i in range(batch_size):
                # Interleave/concatenate instruction and local query text (instruction first)
                text_content = " ".join(
                    [t for t in [instruction, texts[i] if texts else None] if t]
                )

                if is_multimodal:
                    sample = {}
                    # Insert visual keys first to ensure they precede the text tokens
                    if images and images[i] is not None:
                        sample["image"] = images[i]
                    if videos and videos[i] is not None:
                        sample["video"] = videos[i]
                    if text_content:
                        sample["text"] = text_content
                    batched_input.append(sample)
                else:
                    batched_input.append(text_content)

            # Call SentenceTransformer's encode with prompt=None to bypass system-role wrapping
            embeddings = self.model.encode(batched_input, prompt=None, **kwargs)
            all_embeddings.append(embeddings)

        return cast("Array", np.concatenate(all_embeddings, axis=0))


wemm_embedding_2b = ModelMeta(
    loader=WeMMEncoderWrapper,
    loader_kwargs=dict(trust_remote_code=True),
    name="tencent/WeMM-Embedding-2B",
    model_type=["dense"],
    modalities=["text", "image", "video"],
    languages=["zho-Hans", "eng-Latn"],
    open_weights=True,
    revision="df8094e5caf29083d9cac28e96fad6cfbe3ee57f",
    release_date="2026-08-25",
    n_parameters=2_210_000_000,
    n_embedding_parameters=508_063_744,
    memory_usage_mb=None,
    max_tokens=262144,
    embed_dim=[64, 128, 256, 512, 1024, 2048],
    license="apache-2.0",
    reference="https://huggingface.co/tencent/WeMM-Embedding-2B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    citation=WEMM_CITATION,
    public_training_code="https://github.com/Tencent/WeMM-Embedding",
    public_training_data=None,
    training_datasets=None,
    extra_requirements_groups=["wemm"],
)

wemm_embedding_4b = ModelMeta(
    loader=WeMMEncoderWrapper,
    loader_kwargs=dict(trust_remote_code=True),
    name="tencent/WeMM-Embedding-4B",
    model_type=["dense"],
    modalities=["text", "image", "video"],
    languages=["zho-Hans", "eng-Latn"],
    open_weights=True,
    revision="a28b25c5d18cf71ec46b115e06ea79ab00ee4819",
    release_date="2026-08-25",
    n_parameters=4_250_000_000,
    n_embedding_parameters=635_079_680,
    memory_usage_mb=None,
    max_tokens=262144,
    embed_dim=[64, 128, 256, 512, 1024, 2560],
    license="apache-2.0",
    reference="https://huggingface.co/tencent/WeMM-Embedding-4B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    citation=WEMM_CITATION,
    public_training_code="https://github.com/Tencent/WeMM-Embedding",
    public_training_data=None,
    training_datasets=None,
    extra_requirements_groups=["wemm"],
)

wemm_embedding_9b = ModelMeta(
    loader=WeMMEncoderWrapper,
    loader_kwargs=dict(trust_remote_code=True),
    name="tencent/WeMM-Embedding-9B",
    model_type=["dense"],
    modalities=["text", "image", "video"],
    languages=["zho-Hans", "eng-Latn"],
    open_weights=True,
    revision="59f673e3d53c4f71bb364adf344536dc1c2bc95e",
    release_date="2026-08-25",
    n_parameters=9_000_000_000,
    n_embedding_parameters=1_016_127_488,
    memory_usage_mb=None,
    max_tokens=262144,
    embed_dim=[64, 128, 256, 512, 1024, 2048, 4096],
    license="apache-2.0",
    reference="https://huggingface.co/tencent/WeMM-Embedding-9B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    citation=WEMM_CITATION,
    public_training_code="https://github.com/Tencent/WeMM-Embedding",
    public_training_data=None,
    training_datasets=None,
    extra_requirements_groups=["wemm"],
)

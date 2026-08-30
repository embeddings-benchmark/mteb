from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import (
    SentenceTransformerEncoderWrapper,
)
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


class DataLoaderWrapper:
    """Wraps a DataLoader to convert pre-decoded video frame tensors to lists of PIL images."""

    def __init__(self, dataloader: DataLoader[BatchedInput]) -> None:
        self.dataloader = dataloader

    def __len__(self) -> int:
        return len(self.dataloader)

    @property
    def dataset(self) -> Any:
        return self.dataloader.dataset

    @property
    def collate_fn(self) -> Any:
        return self.dataloader.collate_fn

    @collate_fn.setter
    def collate_fn(self, value: Any) -> None:
        self.dataloader.collate_fn = value

    def __iter__(self) -> Any:
        for batch in self.dataloader:
            if "video" in batch:
                batch["video"] = [
                    [
                        to_pil_image(f.permute(2, 0, 1) if f.ndim == 3 and f.shape[-1] == 3 else f)
                        for f in v.cpu()
                    ]
                    if isinstance(v, torch.Tensor)
                    else v
                    for v in batch["video"]
                ]
            yield batch


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
        # A. Wrap the inputs (enables standard video frame decoding)
        wrapped_inputs = DataLoaderWrapper(inputs)

        # B. Check features to see if evaluating a multimodal task (WeMM supports image and video)
        features = wrapped_inputs.dataset.features
        has_video = "video" in features
        is_multimodal = has_video or "image" in features

        # C. Retrieve the public task instruction / prompt
        instruction = self.get_task_instruction(task_metadata, prompt_type)

        if is_multimodal:
            # Setup standard video collator if video modality is active
            if has_video:
                from mteb.models.modality_collators import VideoCollator

                wrapped_inputs.dataloader.collate_fn = VideoCollator(
                    target_sampling_rate=self.target_sampling_rate or 16000,
                    fps=self.fps,
                    max_frames=self.max_frames,
                    num_frames=self.num_frames,
                    max_samples=self.max_samples,
                )

            all_embeddings = []
            for batch in tqdm(wrapped_inputs, desc="Building multimodal embeddings"):
                texts = batch.get("text")
                images = batch.get("image")
                videos = batch.get("video")

                batch_size = len(images) if images else (len(videos) if videos else len(texts))
                batched_input = []

                # Build native interleaved "role: user" chat message lists
                for i in range(batch_size):
                    content = []
                    # 1. Image / Video first
                    if images and images[i] is not None:
                        content.append({"type": "image", "image": images[i]})
                    if videos and videos[i] is not None:
                        content.append({"type": "video", "video": videos[i]})

                    # 2. Task instruction as a separate text item in the content list
                    if instruction:
                        content.append({"type": "text", "text": instruction})

                    # 3. Local dataset query text as a separate text item in the content list
                    if texts and texts[i]:
                        content.append({"type": "text", "text": str(texts[i])})

                    batched_input.append({"role": "user", "content": content})

                # Encode the structured user turns directly with prompt=None
                embeddings = self.model.encode(batched_input, prompt=None, **kwargs)
                all_embeddings.append(embeddings)

            return cast("Array", np.concatenate(all_embeddings, axis=0))

        # D. Fallback to standard text path
        sentences = [text for batch in wrapped_inputs for text in batch["text"]]
        return cast(
            "Array",
            self.model.encode(
                sentences,
                prompt=instruction,
                **kwargs,
            ),
        )


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

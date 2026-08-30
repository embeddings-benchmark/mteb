from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
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


class WeMMEncoderWrapper(AbsEncoder):
    """Custom raw transformers encoder wrapper for WeChat WeMM Embedding models."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device).eval()

        # Ensure padding token is configured for batched execution
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # Store sampling parameters for VideoCollator
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.target_sampling_rate = 16000

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
        # A. Setup the standard MTEB VideoCollator on the inputs if video is present
        features = inputs.dataset.features
        has_video = "video" in features
        is_multimodal = has_video or "image" in features

        instruction = self.get_task_instruction(task_metadata, prompt_type)

        if has_video:
            from mteb.models.modality_collators import VideoCollator

            inputs.collate_fn = VideoCollator(
                target_sampling_rate=self.target_sampling_rate,
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
            )

        all_embeddings = []

        # B. Process the batches sequentially from MTEB's DataLoader directly
        for batch in tqdm(wrapped_inputs := inputs, desc="Encoding batches"):
            texts = batch.get("text")
            images = batch.get("image")
            videos = batch.get("video")

            # C. Convert pre-decoded video frame tensors to lists of PIL images inline
            if videos is not None:
                processed_videos = []
                for video_item in videos:
                    if isinstance(video_item, torch.Tensor):
                        frames = []
                        video_item_cpu = video_item.cpu()
                        for i in range(video_item_cpu.shape[0]):
                            frame = video_item_cpu[i]
                            if frame.ndim == 3 and frame.shape[-1] == 3 and frame.shape[0] != 3:
                                frame = frame.permute(2, 0, 1)
                            frames.append(to_pil_image(frame))
                        processed_videos.append(frames)
                    else:
                        processed_videos.append(video_item)
                videos = processed_videos

            batch_size = len(images) if images else (len(videos) if videos else len(texts))
            batch_messages = []

            # D. Build raw conversational messages (interleaved, strictly inside "user" turn)
            for i in range(batch_size):
                content = []
                # 1. Visual first (preserves Tencent's required ordering)
                if images and images[i] is not None:
                    content.append({"type": "image", "image": images[i]})
                if videos and videos[i] is not None:
                    content.append({"type": "video", "video": videos[i]})

                # 2. Task instruction
                if instruction:
                    content.append({"type": "text", "text": instruction})

                # 3. Local dataset text
                if texts and texts[i]:
                    content.append({"type": "text", "text": str(texts[i])})

                # Fallback to plain text if empty
                if not content and texts:
                    content.append({"type": "text", "text": str(texts[i])})

                batch_messages.append([{"role": "user", "content": content}])

            # E. Render the chat template natively
            text_inputs = [
                self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                for conv in batch_messages
            ]

            # F. Extract vision features and prepare model inputs (padding=True is restored here!)
            if is_multimodal:
                from qwen_vl_utils import process_vision_info

                images_processed, videos_processed, video_kwargs = process_vision_info(
                    batch_messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
                if videos_processed is not None:
                    videos_processed, video_metadata = zip(*videos_processed)
                    videos_processed, video_metadata = list(videos_processed), list(video_metadata)
                else:
                    video_metadata = None

                inputs_pt = self.processor(
                    text=text_inputs,
                    images=images_processed,
                    videos=videos_processed,
                    video_metadata=video_metadata,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs,
                ).to(self.model.device)
            else:
                inputs_pt = self.processor(
                    text=text_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)

            # G. Perform the forward pass and L2 normalize
            with torch.inference_mode():
                embedding_outputs = self.model.embedding(**inputs_pt)
                embedding_outputs = torch.nn.functional.normalize(embedding_outputs, dim=-1)
                all_embeddings.append(embedding_outputs.cpu().float().numpy())

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

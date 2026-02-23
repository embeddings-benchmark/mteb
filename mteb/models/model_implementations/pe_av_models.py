from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm

from mteb._create_dataloaders import AudioCollator
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType
    from mteb.types._encoder_io import AudioInput, TextInput, VideoInput


class PEAudioVisualWrapper(AbsEncoder):
    """MTEB wrapper for PE-AV (Perception Encoder Audio-Visual).

    PE-AV embeds audio, video, audio-video, and text into a joint embedding space.
    Uses the transformers API (PeAudioVideoModel / PeAudioVideoProcessor).

    Video inputs arrive as torchcodec VideoDecoder objects from HF datasets.
    We sample frames using uniform sampling (matching the PE-AV internal
    approach) and pass the decoded frame tensors to the processor.
    """

    def __init__(
        self,
        model_name: str = "facebook/pe-av-large",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_frames: int = 16,
        **kwargs: Any,
    ):
        from transformers import PeAudioVideoModel, PeAudioVideoProcessor

        self.model_name = model_name
        self.device = device
        self.num_frames = num_frames
        self.model = PeAudioVideoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = PeAudioVideoProcessor.from_pretrained(model_name)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def _uniform_sample(self, total: int, n: int) -> list[int]:
        """Uniformly sample n indices from [0, total-1].

        Matches the PE-AV internal sampling strategy.
        """
        if n >= total:
            return list(range(total))
        stride = (total - 1) / (n - 1) if n > 1 else 0
        return [int(round(i * stride)) for i in range(n)]

    def _decode_videos(self, video_decoders: list) -> list[torch.Tensor]:
        """Decode VideoDecoder objects into frame tensors.

        Samples frames uniformly and decodes them using get_frames_at,
        matching the PE-AV internal video loading approach.

        Args:
            video_decoders: List of torchcodec VideoDecoder objects.

        Returns:
            List of frame tensors, each with shape (T, H, W, C).
        """
        decoded = []
        for decoder in video_decoders:
            indices = self._uniform_sample(len(decoder), self.num_frames)
            frames = decoder.get_frames_at(indices=indices).data
            decoded.append(frames)
        return decoded

    def get_text_embeddings(
        self,
        inputs: DataLoader[TextInput],
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get text embeddings aligned to audio-video space."""
        all_embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
            desc="Processing text batches",
        ):
            texts = batch["text"]
            processed = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            processed = {k: v.to(self.device) for k, v in processed.items()}

            with torch.inference_mode(), torch.autocast(
                str(self.device), dtype=torch.bfloat16
            ):
                text_embeds = self.model.get_text_audio_video_embeds(
                    input_ids=processed["input_ids"],
                    attention_mask=processed.get("attention_mask"),
                )
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(text_embeds.cpu().float().numpy())

        return np.vstack(all_embeddings)

    def get_video_embeddings(
        self,
        inputs: DataLoader[VideoInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get video-only embeddings."""
        all_embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
            desc="Processing video batches",
        ):
            videos = self._decode_videos(batch["video"])
            processed = self.processor(
                videos=videos,
                return_tensors="pt",
                padding=True,
            )
            processed = {k: v.to(self.device) for k, v in processed.items()}

            with torch.inference_mode(), torch.autocast(
                str(self.device), dtype=torch.bfloat16
            ):
                video_embeds = self.model.get_video_embeds(
                    pixel_values_videos=processed["pixel_values_videos"],
                    padding_mask_videos=processed.get("padding_mask_videos"),
                )
                video_embeds = video_embeds / video_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(video_embeds.cpu().float().numpy())

        return np.vstack(all_embeddings)

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get audio-only embeddings."""
        all_embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
            desc="Processing audio batches",
        ):
            audio_arrays = [audio["array"] for audio in batch["audio"]]
            processed = self.processor(
                audio=audio_arrays,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            processed = {k: v.to(self.device) for k, v in processed.items()}

            with torch.inference_mode(), torch.autocast(
                str(self.device), dtype=torch.bfloat16
            ):
                audio_embeds = self.model.get_audio_embeds(
                    input_values=processed["input_values"],
                    padding_mask=processed.get("padding_mask"),
                )
                audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(audio_embeds.cpu().float().numpy())

        return np.vstack(all_embeddings)

    def get_audio_video_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get joint audio-video embeddings."""
        all_embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
            desc="Processing audio-video batches",
        ):
            videos = self._decode_videos(batch["video"])
            audio_arrays = [audio["array"] for audio in batch["audio"]]
            processed = self.processor(
                videos=videos,
                audio=audio_arrays,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            processed = {k: v.to(self.device) for k, v in processed.items()}

            with torch.inference_mode(), torch.autocast(
                str(self.device), dtype=torch.bfloat16
            ):
                av_embeds = self.model.get_audio_video_embeds(
                    input_values=processed["input_values"],
                    pixel_values_videos=processed["pixel_values_videos"],
                    padding_mask=processed.get("padding_mask"),
                    padding_mask_videos=processed.get("padding_mask_videos"),
                )
                av_embeds = av_embeds / av_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(av_embeds.cpu().float().numpy())

        return np.vstack(all_embeddings)

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
        has_text = "text" in inputs.dataset.features
        has_video = "video" in inputs.dataset.features
        has_audio = "audio" in inputs.dataset.features

        if has_audio:
            inputs.collate_fn = AudioCollator(self.sampling_rate)

        # Joint audio-video embedding
        if has_video and has_audio and not has_text:
            return self.get_audio_video_embeddings(inputs, **kwargs)

        # Video-only
        if has_video and not has_audio and not has_text:
            return self.get_video_embeddings(inputs, **kwargs)

        # Audio-only
        if has_audio and not has_video and not has_text:
            return self.get_audio_embeddings(inputs, **kwargs)

        # Text-only
        if has_text and not has_video and not has_audio:
            return self.get_text_embeddings(inputs, prompt_type=prompt_type, **kwargs)

        # Mixed modality: fuse embeddings by addition
        embeddings = None

        if has_text:
            text_emb = self.get_text_embeddings(
                inputs, prompt_type=prompt_type, **kwargs
            )
            embeddings = text_emb

        if has_video and has_audio:
            av_emb = self.get_audio_video_embeddings(inputs, **kwargs)
            embeddings = av_emb if embeddings is None else embeddings + av_emb
        elif has_video:
            video_emb = self.get_video_embeddings(inputs, **kwargs)
            embeddings = video_emb if embeddings is None else embeddings + video_emb
        elif has_audio:
            audio_emb = self.get_audio_embeddings(inputs, **kwargs)
            embeddings = audio_emb if embeddings is None else embeddings + audio_emb

        if embeddings is not None:
            return embeddings

        raise ValueError(
            f"No supported modality found in dataset features: {list(inputs.dataset.features.keys())}"
        )


# --- Model Metadata ---

_PE_AV_CITATION = r"""
@misc{vyas2025pushingfrontieraudiovisualperception,
      title={Pushing the Frontier of Audiovisual Perception with Large-Scale Multimodal Correspondence Learning},
      author={Apoorv Vyas and Heng-Jui Chang and Cheng-Fu Yang and Po-Yao Huang and Luya Gao and Julius Richter and Sanyuan Chen and Matt Le and Piotr Dollár and Christoph Feichtenhofer and Ann Lee and Wei-Ning Hsu},
      year={2025},
      eprint={2512.19687},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2512.19687},
}
"""

_PE_AV_COMMON = dict(
    languages=["eng-Latn"],
    release_date="2025-01-01",
    modalities=["audio", "video", "text"],
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    similarity_fn_name="dot",
    use_instructions=False,
    training_datasets=set(),
    citation=_PE_AV_CITATION,
)


pe_av_small_16_frame = ModelMeta(
    loader=PEAudioVisualWrapper,
    name="facebook/pe-av-small-16-frame",
    revision="9f888eea95c83622212bb742e91bf01d3b46fe96",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    reference="https://huggingface.co/facebook/pe-av-small-16-frame",
    **_PE_AV_COMMON,
)

pe_av_base_16_frame = ModelMeta(
    loader=PEAudioVisualWrapper,
    name="facebook/pe-av-base-16-frame",
    revision="3fd870e60a1099fb99367b240d41926e183d7112",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    reference="https://huggingface.co/facebook/pe-av-base-16-frame",
    **_PE_AV_COMMON,
)

pe_av_large_16_frame = ModelMeta(
    loader=PEAudioVisualWrapper,
    name="facebook/pe-av-large-16-frame",
    revision="446d823f089b1301a0ff37175ab283b6927db757",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    reference="https://huggingface.co/facebook/pe-av-large-16-frame",
    **_PE_AV_COMMON,
)

pe_av_small = ModelMeta(
    loader=PEAudioVisualWrapper,
    name="facebook/pe-av-small",
    revision="dd050762bb9704ae9cd996ca45532a98f81d817e",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    reference="https://huggingface.co/facebook/pe-av-small",
    **_PE_AV_COMMON,
)

pe_av_base = ModelMeta(
    loader=PEAudioVisualWrapper,
    name="facebook/pe-av-base",
    revision="1c4c329b78d80a18d860ebfb05daebff4bc44518",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    reference="https://huggingface.co/facebook/pe-av-base",
    **_PE_AV_COMMON,
)

pe_av_large = ModelMeta(
    loader=PEAudioVisualWrapper,
    name="facebook/pe-av-large",
    revision="0d24878d4107d64bef49e53602fc34ce6f94f6d8",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    reference="https://huggingface.co/facebook/pe-av-large",
    **_PE_AV_COMMON,
)

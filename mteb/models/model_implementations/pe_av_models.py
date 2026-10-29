from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import VideoCollator
from mteb.types import OutputDType, PromptType

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput
    from mteb.types._encoder_io import AudioInput, TextInput, VideoInput


def _other_side_modalities(
    task_metadata: TaskMetadata, prompt_type: PromptType | None
) -> set[str]:
    """Modalities of the side this input is compared with."""
    if prompt_type is None:
        return set(task_metadata.modalities)
    other_side = (
        PromptType.document if prompt_type == PromptType.query else PromptType.query
    )
    return set(task_metadata.get_modalities(other_side))


class PEAudioVisualWrapper(AbsEncoder):
    """MTEB wrapper for PE-AV (Perception Encoder Audio-Visual).

    PE-AV embeds audio, video, audio-video, and text into a joint embedding space.
    Uses the transformers API (PeAudioVideoModel / PeAudioVideoProcessor).

    Video inputs arrive as pre-decoded frame tensors via the FramesCollator.
    Audio is read from the separate audio column in each batch.
    """

    def __init__(
        self,
        model_name: str = "facebook/pe-av-large",
        revision: str | None = None,
        device: str | None = None,
        # PE-AV processes every frame it is given (do_sample_frames=false; the
        # paper samples at 30 fps); fps=2 with at most 64 frames is an mteb cap
        # for memory
        # https://huggingface.co/facebook/pe-av-large/blob/main/video_preprocessor_config.json
        fps: float | None = 2.0,
        max_frames: int | None = 64,
        num_frames: int | None = None,
        # full clips, as in Meta's eval, up to the rotary table: 10000 positions
        # (max_position_embeddings) minus CLS, at one per 1920 samples (~400 s);
        # Meta's code fails past it
        # https://github.com/facebookresearch/perception_models/blob/main/core/audio_visual_encoder/transformer.py
        max_samples: int | None = (10_000 - 1) * 1920,
        **kwargs: Any,
    ):
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        from transformers import PeAudioVideoModel, PeAudioVideoProcessor

        self.model_name = model_name
        self.device = device
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.max_samples = max_samples
        self.model = PeAudioVideoModel.from_pretrained(
            model_name, revision=revision
        ).to(self.device)
        self.model.eval()
        # Meta's code pools the CLS token before ModernBERT's final norm
        # https://github.com/facebookresearch/perception_models/blob/main/core/audio_visual_encoder/pe.py
        self.model.text_model.final_norm = torch.nn.Identity()
        self.processor = PeAudioVideoProcessor.from_pretrained(
            model_name, revision=revision
        )
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def _text_cls(self, texts: list[str]) -> torch.Tensor:
        processed = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        processed = {k: v.to(self.device) for k, v in processed.items()}
        text_outputs = self.model.text_model(
            input_ids=processed["input_ids"],
            attention_mask=processed.get("attention_mask"),
        )
        return text_outputs.last_hidden_state[:, 0]

    def get_text_embeddings(
        self,
        inputs: DataLoader[TextInput],
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get text embeddings in the space of the task's other modality."""
        import torch

        # Meta's eval embeds text with the head of the modality it is compared with
        # (encode_video_text / encode_audio_text / encode_audio_video_text)
        other = _other_side_modalities(task_metadata, prompt_type) - {"text"}
        if other == {"video"}:
            head = self.model.video_model.text_video_head
        elif other == {"audio"}:
            head = self.model.audio_model.text_audio_head
        else:
            head = self.model.text_audio_video_head
        with torch.inference_mode():
            all_embeddings = []

            for batch in tqdm(
                inputs,
                disable=not show_progress_bar,
                desc="Processing text batches",
            ):
                text_embeds = head(self._text_cls(batch["text"]))
                text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(text_embeds.cpu().float().numpy())

            return np.vstack(all_embeddings)

    def get_video_embeddings(
        self,
        inputs: DataLoader[VideoInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get video-only embeddings."""
        import torch

        with torch.inference_mode():
            all_embeddings = []

            for batch in tqdm(
                inputs,
                disable=not show_progress_bar,
                desc="Processing video batches",
            ):
                videos = list(batch["video"])
                processed = self.processor(
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                )
                processed = {k: v.to(self.device) for k, v in processed.items()}

                video_out = self.model.video_model.video_encoder(
                    pixel_values_videos=processed["pixel_values_videos"],
                    padding_mask_videos=processed.get("padding_mask_videos"),
                )
                video_embeds = self.model.video_model.video_head(
                    video_out.pooler_output
                )
                video_embeds /= video_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(video_embeds.cpu().float().numpy())

            return np.vstack(all_embeddings)

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get audio-only embeddings."""
        import torch

        with torch.inference_mode():
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

                audio_embeds = self.model.get_audio_embeds(
                    input_values=processed["input_values"],
                    padding_mask=processed.get("padding_mask"),
                )
                audio_embeds /= audio_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(audio_embeds.cpu().float().numpy())

            return np.vstack(all_embeddings)

    def get_audio_video_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get joint audio-video embeddings."""
        import torch

        with torch.inference_mode():
            all_embeddings = []

            for batch in tqdm(
                inputs,
                disable=not show_progress_bar,
                desc="Processing audio-video batches",
            ):
                videos = list(batch["video"])
                audio_arrays = [audio["array"] for audio in batch["audio"]]
                processed = self.processor(
                    videos=videos,
                    audio=audio_arrays,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding=True,
                )
                processed = {k: v.to(self.device) for k, v in processed.items()}

                av_output = self.model.get_audio_video_embeds(
                    input_values=processed["input_values"],
                    pixel_values_videos=processed["pixel_values_videos"],
                    padding_mask=processed.get("padding_mask"),
                    padding_mask_videos=processed.get("padding_mask_videos"),
                )
                av_embeds = av_output.audio_video_embeds
                av_embeds /= av_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(av_embeds.cpu().float().numpy())

            return np.vstack(all_embeddings)

    def get_media_plus_text_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        use_video: bool,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Embed video+text or audio+text with the fused head, as Meta's encode_*_plus_text."""
        import torch

        all_embeddings = []
        with torch.inference_mode():
            for batch in tqdm(
                inputs,
                disable=not show_progress_bar,
                desc="Processing media+text batches",
            ):
                text_cls = self._text_cls(batch["text"])
                if use_video:
                    processed = self.processor(
                        videos=list(batch["video"]),
                        return_tensors="pt",
                        padding=True,
                    )
                    processed = {k: v.to(self.device) for k, v in processed.items()}
                    media = self.model.video_model.video_encoder(
                        pixel_values_videos=processed["pixel_values_videos"],
                        padding_mask_videos=processed.get("padding_mask_videos"),
                    ).pooler_output
                    head = self.model.video_plus_text_head
                else:
                    processed = self.processor(
                        audio=[a["array"] for a in batch["audio"]],
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                        padding=True,
                    )
                    processed = {k: v.to(self.device) for k, v in processed.items()}
                    media = self.model.audio_model.audio_encoder(
                        input_values=processed["input_values"],
                        padding_mask=processed.get("padding_mask"),
                    ).pooler_output
                    head = self.model.audio_plus_text_head
                # media first, then text (pe.py encode_*_plus_text)
                embeds = head(torch.cat([media, text_cls], dim=-1))
                embeds /= embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeds.cpu().float().numpy())
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

        inputs.collate_fn = VideoCollator(
            target_sampling_rate=self.sampling_rate,
            fps=self.fps,
            max_frames=self.max_frames,
            num_frames=self.num_frames,
            max_samples=self.max_samples,
        )

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
            return self.get_text_embeddings(
                inputs, prompt_type=prompt_type, task_metadata=task_metadata, **kwargs
            )

        # video+text -> audio / audio+text -> video: the fused heads are only
        # trained against that other modality (PeAudioVideoModel.forward scores
        # audio_plus_text @ video.T and video_plus_text @ audio.T)
        fused_target = {"audio"} if has_video else {"video"}
        if (
            has_text
            and has_video != has_audio
            and _other_side_modalities(task_metadata, prompt_type) == fused_target
        ):
            return self.get_media_plus_text_embeddings(
                inputs, use_video=has_video, **kwargs
            )

        # Other mixed inputs (no dedicated head): fuse embeddings by addition
        embeddings = None

        if has_text:
            text_emb = self.get_text_embeddings(
                inputs, prompt_type=prompt_type, task_metadata=task_metadata, **kwargs
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

_PE_AV_TRAINING = {
    "AudioCapsA2TRetrieval",
    "AudioCapsT2ARetrieval",
    "AudioCapsAVA2VRetrieval",
    "AudioCapsAVAT2VRetrieval",
    "AudioCapsAVT2VRetrieval",
    "AudioCapsAVT2VARetrieval",
    "AudioCapsAVV2ARetrieval",
    "AudioCapsAVV2TRetrieval",
    "AudioCapsAVVA2TRetrieval",
    "AudioCapsAVVT2ARetrieval",
    "VALOR32KA2VRetrieval",
    "VALOR32KAT2VRetrieval",
    "VALOR32KT2VARetrieval",
    "VALOR32KT2VRetrieval",
    "VALOR32KV2ARetrieval",
    "VALOR32KV2TRetrieval",
    "VALOR32KVA2TRetrieval",
    "VALOR32KVT2ARetrieval",
    "VGGSoundV",
    "VGGSoundVA",
    "VGGSoundVideoZeroshot",
    "VGGSoundVideoAudioZeroshot",
    "VGGSoundAVA2VRetrieval",
    "VGGSoundAVAT2VRetrieval",
    "VGGSoundAVT2VRetrieval",
    "VGGSoundAVT2VARetrieval",
    "VGGSoundAVV2ARetrieval",
    "VGGSoundAVV2TRetrieval",
    "VGGSoundAVVA2TRetrieval",
    "VGGSoundAVVT2ARetrieval",
    "Kinetics400V",
    "Kinetics400VA",
    "Kinetics400ZeroShot",
    "Kinetics400VAZeroShot",
    "Kinetics700V",
    "Kinetics700VA",
    "Kinetics700VZeroShot",
    "Kinetics700VAZeroShot",
    "HMDB51Classification",
    "HMDB51Clustering",
    "HMDB51ZeroShot",
    "MSRVTTA2V",
    "MSRVTTAT2V",
    "MSRVTTT2V",
    "MSRVTTT2VA",
    "MSRVTTV2A",
    "MSRVTTV2T",
    "MSRVTTVA2T",
    "MSRVTTVT2A",
    "MSVDT2VRetrieval",
    "MSVDV2TRetrieval",
    "ActivityNetCaptionsT2VRetrieval",
    "ActivityNetCaptionsV2TRetrieval",
    "DiDeMoA2VRetrieval",
    "DiDeMoAT2VRetrieval",
    "DiDeMoT2VRetrieval",
    "DiDeMoT2VARetrieval",
    "DiDeMoV2ARetrieval",
    "DiDeMoV2TRetrieval",
    "DiDeMoVA2TRetrieval",
    "DiDeMoVT2ARetrieval",
}

_PE_AV_COMMON = dict(
    languages=["eng-Latn"],
    release_date="2025-12-22",
    modalities=["audio", "video", "text"],
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    similarity_fn_name="dot",
    use_instructions=False,
    training_datasets=_PE_AV_TRAINING,
    citation=_PE_AV_CITATION,
    # PeAudioVideoModel ships with transformers 5
    extra_requirements_groups=["transformers-v5"],
)


pe_av_small_16_frame = ModelMeta(
    loader=PEAudioVisualWrapper,
    name="facebook/pe-av-small-16-frame",
    revision="9f888eea95c83622212bb742e91bf01d3b46fe96",
    n_parameters=846_991_308,
    n_embedding_parameters=51576832,
    memory_usage_mb=3231,
    max_tokens=512,
    embed_dim=1024,
    reference="https://huggingface.co/facebook/pe-av-small-16-frame",
    loader_kwargs=dict(fps=None, num_frames=16),
    **_PE_AV_COMMON,
    output_dtypes=OutputDType.BF16,
)

pe_av_base_16_frame = ModelMeta(
    loader=PEAudioVisualWrapper,
    name="facebook/pe-av-base-16-frame",
    revision="3fd870e60a1099fb99367b240d41926e183d7112",
    n_parameters=1_033_701_580,
    n_embedding_parameters=51576832,
    memory_usage_mb=3943,
    max_tokens=512,
    embed_dim=1024,
    reference="https://huggingface.co/facebook/pe-av-base-16-frame",
    loader_kwargs=dict(fps=None, num_frames=16),
    **_PE_AV_COMMON,
    output_dtypes=OutputDType.BF16,
)

pe_av_large_16_frame = ModelMeta(
    loader=PEAudioVisualWrapper,
    name="facebook/pe-av-large-16-frame",
    revision="446d823f089b1301a0ff37175ab283b6927db757",
    n_parameters=2_234_700_748,
    n_embedding_parameters=51576832,
    memory_usage_mb=8525,
    max_tokens=512,
    embed_dim=1024,
    reference="https://huggingface.co/facebook/pe-av-large-16-frame",
    loader_kwargs=dict(fps=None, num_frames=16),
    **_PE_AV_COMMON,
    output_dtypes=OutputDType.BF16,
)

pe_av_small = ModelMeta(
    loader=PEAudioVisualWrapper,
    name="facebook/pe-av-small",
    revision="dd050762bb9704ae9cd996ca45532a98f81d817e",
    n_parameters=846_991_308,
    n_embedding_parameters=51576832,
    memory_usage_mb=3231,
    max_tokens=512,
    embed_dim=1024,
    reference="https://huggingface.co/facebook/pe-av-small",
    **_PE_AV_COMMON,
    output_dtypes=OutputDType.BF16,
)

pe_av_base = ModelMeta(
    loader=PEAudioVisualWrapper,
    name="facebook/pe-av-base",
    revision="1c4c329b78d80a18d860ebfb05daebff4bc44518",
    n_parameters=1_033_701_580,
    n_embedding_parameters=51576832,
    memory_usage_mb=3943,
    max_tokens=512,
    embed_dim=1024,
    reference="https://huggingface.co/facebook/pe-av-base",
    **_PE_AV_COMMON,
    output_dtypes=OutputDType.BF16,
)

pe_av_large = ModelMeta(
    loader=PEAudioVisualWrapper,
    name="facebook/pe-av-large",
    revision="0d24878d4107d64bef49e53602fc34ce6f94f6d8",
    n_parameters=2_234_700_748,
    n_embedding_parameters=51576832,
    memory_usage_mb=8525,
    max_tokens=512,
    embed_dim=1024,
    reference="https://huggingface.co/facebook/pe-av-large",
    **_PE_AV_COMMON,
    output_dtypes=OutputDType.BF16,
)

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator, VideoCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

# OmniRet's Qwen-Audio tower expects 16 kHz mono; its whisper-style frontend
# consumes a 30 second window.
OMNIRET_SAMPLING_RATE = 16_000
OMNIRET_MAX_AUDIO_SECONDS = 30
# omniret.config.OmniRetModelConfig.max_video_frames is 8 and the model
# re-subsamples above that, so sample 8 at the collator to save decode work.
OMNIRET_MAX_VIDEO_FRAMES = 8


class OmniRetWrapper(AbsEncoder):
    """Wrapper for OmniRet, an instruction-aware text/image/audio/video retriever.

    Every input maps to one L2-normalized 4096-d vector via Attention Sliced
    Wasserstein Pooling, so cosine similarity is the scoring function.

    This calls ``OmniRetModel.encode_raw_media_batch`` and ``encode_batch``
    directly rather than ``OmniRetEmbedder.process``, because ``process`` only
    accepts filesystem paths or HTTP URLs and would force a temp-file write per
    item. The two-call path accepts in-memory PIL images, frame lists, and audio
    arrays, which is what mteb's collators already produce.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        max_audio_length_seconds: int = OMNIRET_MAX_AUDIO_SECONDS,
        num_frames: int = OMNIRET_MAX_VIDEO_FRAMES,
        **kwargs: Any,
    ) -> None:
        from huggingface_hub import snapshot_download

        try:
            from omniret import OmniRetEmbedder
        except ImportError as error:
            raise ImportError(
                "OmniRetWrapper requires the `omniret` package, which is not on PyPI. "
                "Install it into a Python 3.10 CUDA 12.8 environment with "
                "`uv pip install --no-deps git+https://github.com/hmchuong/OmniRet` "
                "after installing torch 2.9.1+cu128 from the pytorch cu128 index."
            ) from error

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.sampling_rate = OMNIRET_SAMPLING_RATE
        self.max_samples = int(max_audio_length_seconds * self.sampling_rate)

        # OmniRetEmbedder does not take a revision, so resolve the snapshot
        # ourselves and hand it the pinned local directory. Base encoder
        # revisions are pinned separately inside the checkpoint's config.json.
        model_dir = snapshot_download(repo_id=model_name, revision=revision)

        # sdpa keeps flash-attn out of the dependency set entirely; OmniRet only
        # forwards this to the text tower and already hardcodes sdpa elsewhere.
        embedder = OmniRetEmbedder(
            model_dir,
            device=self.device,
            attn_implementation=kwargs.pop("attn_implementation", "sdpa"),
            **kwargs,
        )
        self.model = embedder.model
        self.model.eval()

    @staticmethod
    def _row_modality(video: Any, audio: Any) -> str:
        """Pick the single media slot for a row. OmniRet allows at most one."""
        if video is not None:
            return "video"
        if audio is not None:
            return "audio"
        # Text-only rows are labelled "image" and carry no media, matching the
        # default in OmniRetEmbedder.process.
        return "image"

    @staticmethod
    def _format_text(
        instruction: str, modality: str, text: str, has_media: bool
    ) -> str:
        """Reproduce omniret.embedding._format_input exactly.

        The '<image>' / '<video>' / '<audio>' placeholder is where
        _expand_media_placeholders splices media tokens, and the 'Query:\\n'
        marker is what _without_instruction_prefix keys off. Both strings are
        part of the model's trained input format, so do not reformat them.
        """
        pieces = []
        if has_media:
            pieces.append(f"{modality.title()}: <{modality}>")
        if text:
            pieces.append(text)
        body = "\n".join(pieces)
        return f"Instruct: {instruction}\nQuery:\n{body}" if instruction else body

    @staticmethod
    def _coerce_video(item: Any) -> Any:
        """Reduce mteb's video payload to the PIL frame list _load_video returns.

        VideoCollator yields a uint8 NCHW torch.Tensor from torchcodec, but
        OmniRet's _raw_video_frames evaluates ``video or []``, which raises on a
        multi-element tensor. Its vision path feeds frames straight to the SigLIP
        image processor, so PIL RGB is the shape that matches the reference path.
        """
        from PIL import Image

        if item is None or isinstance(item, (list, tuple)):
            return item
        if isinstance(item, dict):
            return item.get("video", item.get("image"))
        if not torch.is_tensor(item):
            return item

        frames = item.detach().cpu()
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        if frames.shape[1] in {1, 3} and frames.shape[-1] not in {1, 3}:
            frames = frames.permute(0, 2, 3, 1)
        if frames.dtype != torch.uint8:
            scale = 255.0 if frames.max() <= 1.0 else 1.0
            frames = (frames.float() * scale).clamp(0, 255).to(torch.uint8)
        return [Image.fromarray(frame.numpy()).convert("RGB") for frame in frames]

    @staticmethod
    def _coerce_audio(item: Any) -> Any:
        """Reduce mteb's audio payload to the 1-D float32 waveform _load_wav returns.

        OmniRet's _prepare_qwen_audio reads ``item["audio"]``, so mteb's
        ``{"array": ..., "sampling_rate": ...}`` resolves to None and is silently
        replaced with N_SAMPLES of zeros. It also treats any 2-D tensor as a
        precomputed log-mel spectrogram, so a (channels, samples) array would be
        misread rather than raising.
        """
        if isinstance(item, dict):
            item = item.get("array", item.get("audio"))
        if item is None:
            return None
        waveform = (
            item.detach().to(torch.float32)
            if torch.is_tensor(item)
            else torch.as_tensor(item, dtype=torch.float32)
        )
        if waveform.ndim > 2:
            waveform = waveform.reshape(waveform.shape[0], -1)
        if waveform.ndim == 2:
            # mteb and torchaudio use (channels, samples); downmix to mono.
            waveform = waveform.mean(dim=0)
        return waveform.flatten()

    def _prepare_batch(
        self, batch: BatchedInput, instruction: str
    ) -> tuple[list[str], list[str], list[Any]]:
        """Split a collated batch into aligned text, modality, and media lists."""
        columns = {
            key: batch.get(key) or [] for key in ("text", "image", "audio", "video")
        }
        size = max(len(value) for value in columns.values())

        formatted: list[str] = []
        modalities: list[str] = []
        media: list[Any] = []
        for index in range(size):
            row = {
                key: value[index] if index < len(value) else None
                for key, value in columns.items()
            }
            modality = self._row_modality(row["video"], row["audio"])
            item = row[modality]
            if modality == "audio":
                item = self._coerce_audio(item)
            elif modality == "video":
                item = self._coerce_video(item)
            media.append(item)
            modalities.append(modality)
            formatted.append(
                self._format_text(
                    instruction,
                    modality,
                    str(row["text"] or "").strip(),
                    item is not None,
                )
            )
        return formatted, modalities, media

    @torch.no_grad()
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
        features = inputs.dataset.features
        if "video" in features:
            inputs.collate_fn = VideoCollator(
                target_sampling_rate=self.sampling_rate,
                num_frames=self.num_frames,
                max_samples=self.max_samples,
            )
        elif "audio" in features:
            inputs.collate_fn = AudioCollator(
                target_sampling_rate=self.sampling_rate,
                max_samples=self.max_samples,
            )

        instruction = self.get_task_instruction(task_metadata, prompt_type)

        all_embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, desc="Encoding"):
            formatted, modalities, media = self._prepare_batch(batch, instruction)

            raw = None
            if any(item is not None for item in media):
                raw = self.model.encode_raw_media_batch(media, modalities)
            tokens, media_mask = raw if isinstance(raw, tuple) else (raw, None)

            embeddings, _ = self.model.encode_batch(
                formatted,
                tokens,
                modalities,
                media_mask,
                exclude_instruction_prefix=bool(instruction),
            )
            # process() normalizes as its final step; match it so cosine
            # scoring and any dot-product consumer agree.
            all_embeddings.append(F.normalize(embeddings.float(), dim=-1).cpu())

        return torch.cat(all_embeddings, dim=0)


omniret = ModelMeta(
    loader=OmniRetWrapper,
    name="chuonghm/OmniRet",
    revision="13a87d6dead2ecabad5f181c8dfc0fe0cb4df314",
    release_date="2026-08-07",
    languages=["eng-Latn"],
    n_parameters=2_697_128_513,
    memory_usage_mb=5148,
    n_embedding_parameters=232_928_256,
    max_tokens=1024,  # omniret.config text_max_length
    embed_dim=4096,
    license="https://github.com/hmchuong/OmniRet/blob/main/NOTICE",
    open_weights=True,
    public_training_code="https://github.com/hmchuong/OmniRet",
    public_training_data="https://huggingface.co/datasets/chuonghm/OmniRet-train",
    framework=["PyTorch", "safetensors"],
    reference="https://huggingface.co/chuonghm/OmniRet",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=None,  # TODO map the M-BEIR/UniIR + audio corpora
    adapted_from="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    superseded_by=None,
    modalities=["text", "image", "audio", "video"],
    model_type=["dense"],
    extra_requirements_groups=["omniret"],
    contacts=["hubielu"],
    citation="""
@inproceedings{huynh2026omniret,
    title={Efficient and High-Fidelity Omni Modality Retrieval},
    author={Huynh, Chuong and Luong, Manh and Shrivastava, Abhinav},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2026}
}""",
)

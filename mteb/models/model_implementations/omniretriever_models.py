from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import VideoCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

WAVE_BASE_MODEL = "tsinghua-ee/WAVE-7B"
WAVE_BASE_REVISION = "7d51cdaecfaabb9c529a447249cd4c2a6df8ce5b"


class OmniRetrieverWrapper(AbsEncoder):
    """MTEB wrapper for OmniRetriever-7B (LoRA adapter on the WAVE-7B backbone).

    OmniRetriever-7B is published as a PEFT adapter only. The backbone
    (``tsinghua-ee/WAVE-7B``, a Qwen2.5-Omni thinker extended with a BEATs audio
    encoder and an all-layer fusion head) is self-contained on the Hub: the
    BEATs weights, the ``classify_linear`` fusion head and the ``beats_ln`` /
    ``beats_proj`` audio adaptors all ship inside its shards, and its
    ``config.json`` already sets ``classify_type="all_layer"``. No ``WAVE_HOME``
    directory or manually downloaded checkpoint is required.

    The adapter's ``modules_to_save`` (``classify_linear``, ``beats_ln``,
    ``beats_proj``) are restored by stock ``PeftModel.from_pretrained``; the
    saved key layout is exactly what PEFT emits and re-maps on load.

    Embeddings are the backbone's ``mllm_embeds``: the last-token hidden state
    of every one of the 28 decoder layers, concatenated (28 x 3584 = 100352) and
    projected by ``classify_linear`` to 3584 dims, then L2-normalised here (the
    model does not normalise internally).

    All preprocessing constants below come from the official training launcher
    (``training/train.sh``) and data pipeline (``training/qwenvl/data/data_qwen.py``)
    in https://github.com/yunzeliu/Omni-Retriever, which define the released
    model's behaviour.
    """

    AUDIO_SAMPLING_RATE = 16_000
    MIN_AUDIO_SEC = 1
    MAX_AUDIO_SEC = 300
    POSITION_ID_PER_SECONDS = 25
    EMBED_DIM = 3584

    def __init__(
        self,
        model_name: str,
        revision: str,
        *,
        base_model_name_or_path: str = WAVE_BASE_MODEL,
        base_model_revision: str = WAVE_BASE_REVISION,
        device: str | None = None,
        num_frames: int = 8,
        pixels: int = 50_176,
        video_batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        from peft import PeftModel
        from transformers import AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # The backbone was trained in bf16 and its BEATs branch runs under a
        # hardcoded ``torch.amp.autocast("cuda", torch.float16)`` in the remote
        # code, so fp32 on CPU would silently diverge for any audio input.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "OmniRetriever-7B requires a CUDA device: the WAVE-7B backbone runs "
                "its BEATs audio encoder under a hardcoded CUDA autocast block."
            )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "OmniRetriever-7B requires bfloat16 support, which this GPU lacks."
            )

        backbone = AutoModel.from_pretrained(
            base_model_name_or_path,
            revision=base_model_revision,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **kwargs,
        )
        self.model = PeftModel.from_pretrained(backbone, model_name, revision=revision)
        self.model = self.model.to(self.device).eval()

        self.processor = self._load_processor(
            base_model_name_or_path, base_model_revision
        )
        self.tokenizer = self.processor.tokenizer

        image_processor = self.processor.image_processor
        image_processor.max_pixels = pixels
        image_processor.min_pixels = pixels
        if getattr(image_processor, "size", None) is not None:
            image_processor.size["longest_edge"] = pixels
            image_processor.size["shortest_edge"] = pixels

        self.video_batch_size = video_batch_size
        self.max_audio_samples = self.MAX_AUDIO_SEC * self.AUDIO_SAMPLING_RATE
        self.collator = VideoCollator(
            target_sampling_rate=self.AUDIO_SAMPLING_RATE,
            num_frames=num_frames,
            max_samples=self.max_audio_samples,
        )

    @staticmethod
    def _load_processor(base_model_name_or_path: str, revision: str) -> Any:  # noqa: ANN401 -- class resolved dynamically from the repo's auto_map
        """Load WAVE-7B's multimodal processor.

        The backbone repo ships no ``processor_config.json``, so
        ``AutoProcessor.from_pretrained`` silently returns the bare
        ``Qwen2TokenizerFast`` instead of honouring the ``AutoProcessor`` entry in
        ``config.json``'s ``auto_map``. Resolve that entry explicitly, and only
        fall back to ``AutoProcessor`` if the repo ever gains a processor config.
        """
        from transformers import AutoConfig, AutoProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config = AutoConfig.from_pretrained(
            base_model_name_or_path, revision=revision, trust_remote_code=True
        )
        reference = (getattr(config, "auto_map", None) or {}).get("AutoProcessor")
        if reference is not None:
            processor_cls = get_class_from_dynamic_module(
                reference, base_model_name_or_path, revision=revision
            )
            return processor_cls.from_pretrained(
                base_model_name_or_path, revision=revision, trust_remote_code=True
            )

        processor = AutoProcessor.from_pretrained(
            base_model_name_or_path, revision=revision, trust_remote_code=True
        )
        if not hasattr(processor, "replace_multimodal_special_tokens"):
            raise TypeError(
                f"Expected a Qwen2.5-Omni style processor for "
                f"{base_model_name_or_path}, got {type(processor).__name__}."
            )
        return processor

    @staticmethod
    def _frames_to_thwc(frames: torch.Tensor) -> np.ndarray:
        """Convert torchcodec ``(T, C, H, W)`` uint8 frames to ``(T, H, W, C)``."""
        return frames.permute(0, 2, 3, 1).contiguous().numpy()

    def _build_prompt(self, caption: str | None, has_video: bool, has_audio: bool):
        """Build the bare user-turn prompt for a modality combination.

        Mirrors ``_prepare_submodal_input`` in ``data_qwen.py``: a single media
        tag (video takes precedence, with audio folded in via
        ``use_audio_in_video``), then the text part, with the chat template
        stripped down to the user content.
        """
        content: list[dict[str, Any]] = []
        if has_video:
            content.append({"type": "video"})
        elif has_audio:
            content.append({"type": "audio"})

        if caption is not None:
            content.append({"type": "text", "text": caption})
        else:
            content.append({"type": "text", "text": "Please describe the video."})

        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=False,
            tokenize=False,
        )
        return text[0].split("<|im_start|>user\n")[-1].strip()

    def _prepare_audio(
        self, audios: list[Any]
    ) -> tuple[dict[str, torch.Tensor], list[torch.Tensor], list[int]]:
        """Build Whisper features, raw BEATs waveforms and placeholder counts."""
        if not audios:
            return {}, [], []

        minimum = self.MIN_AUDIO_SEC * self.AUDIO_SAMPLING_RATE
        waveforms = []
        for audio in audios:
            waveform = np.asarray(audio["array"], dtype=np.float32)
            if waveform.shape[-1] < minimum:
                waveform = np.pad(waveform, (0, minimum - waveform.shape[-1]))
            waveforms.append(waveform)
        raw_wavs = [torch.from_numpy(w) for w in waveforms]
        features = self.processor.feature_extractor(
            waveforms,
            sampling_rate=self.AUDIO_SAMPLING_RATE,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        feature_attention_mask = features.pop("attention_mask")
        audio_inputs = {
            "input_features": features.pop("input_features"),
            "feature_attention_mask": feature_attention_mask,
        }
        # Mirrors Qwen2_5OmniProcessor.__call__: conv then pooling downsampling.
        input_lengths = (feature_attention_mask.sum(-1) - 1) // 2 + 1
        audio_lengths = list((input_lengths - 2) // 2 + 1)
        return audio_inputs, raw_wavs, audio_lengths

    def _prepare_video(
        self, videos: list[Any], durations: list[float | None]
    ) -> tuple[dict[str, Any], list[float]]:
        """Build video pixel inputs and the per-grid second offsets."""
        if not videos:
            return {}, []

        frames = [self._frames_to_thwc(v) for v in videos]
        processed = dict(
            self.processor.image_processor(
                images=None, videos=frames, return_tensors="pt"
            )
        )
        temporal_patch_size = self.processor.image_processor.temporal_patch_size
        second_per_grid = []
        for i, frame_stack in enumerate(frames):
            duration = durations[i] if i < len(durations) else None
            # fps = sampled frames / clip duration, as in data_qwen.process_video
            fps = (len(frame_stack) / duration) if duration else 1.0
            second_per_grid.append(temporal_patch_size / fps)
        # Must be a tensor: the backbone does ``2.0 * second_per_grids`` and
        # indexes it per video, both of which fail on a plain list. The stock
        # processor only gets a tensor via ``BatchFeature(tensor_type="pt")``.
        processed["video_second_per_grid"] = torch.tensor(
            second_per_grid, dtype=torch.float32
        )
        return processed, second_per_grid

    def _tokenize(
        self,
        prompts: list[str],
        *,
        audio_lengths: list[int],
        video_grid_thw: torch.Tensor,
        second_per_grid: list[float],
        use_audio_in_video: bool,
        seconds_per_chunk: float | None,
    ) -> dict[str, torch.Tensor]:
        """Expand media placeholders and tokenize with left padding."""
        prompts = self.processor.replace_multimodal_special_tokens(
            prompts,
            iter(audio_lengths),
            iter([]),  # no image inputs
            iter(video_grid_thw),
            video_second_per_grid=iter(second_per_grid),
            use_audio_in_video=use_audio_in_video,
            position_id_per_seconds=self.POSITION_ID_PER_SECONDS,
            seconds_per_chunk=seconds_per_chunk,
        )
        if audio_lengths:
            # BEATs features are interleaved 1:1 with the Whisper features, so
            # every audio placeholder is doubled (data_qwen.py, use_beats path).
            prompts = [p.replace("<|AUDIO|>", "<|AUDIO|><|AUDIO|>") for p in prompts]

        # Left padding is required: pooling reads the last token of each row.
        return self.tokenizer(
            prompts, padding=True, padding_side="left", return_tensors="pt"
        )

    @staticmethod
    def _unpack(batch: BatchedInput) -> tuple[list, list, list, list]:
        """Split a batch into text / video / audio / duration columns, validating it."""
        texts = batch.get("text") or []
        videos = batch.get("video") or []
        audios = batch.get("audio") or []
        if not (texts or videos or audios):
            if not batch:
                raise ValueError("OmniRetriever received an empty batch.")
            raise ValueError(
                "OmniRetriever supports at least one of text, video or audio, but the "
                f"batch only carries {sorted(batch)}. The released adapter has no "
                "image-only path."
            )
        return texts, videos, audios, batch.get("video_duration") or []

    def _encode_batch(self, batch: BatchedInput) -> torch.Tensor:
        texts, videos, audios, durations = self._unpack(batch)
        # data_qwen.py folds audio into the video stream when both are present.
        use_audio_in_video = bool(videos) and bool(audios)

        audio_inputs, raw_wavs, audio_lengths = self._prepare_audio(audios)
        video_inputs, second_per_grid = self._prepare_video(videos, durations)

        prompts = [
            self._build_prompt(
                texts[i] if i < len(texts) else None, bool(videos), bool(audios)
            )
            for i in range(max(len(texts), len(videos), len(audios)))
        ]
        tokenized = self._tokenize(
            prompts,
            audio_lengths=audio_lengths,
            video_grid_thw=video_inputs.get("video_grid_thw", []),
            second_per_grid=second_per_grid,
            use_audio_in_video=use_audio_in_video,
            # data_qwen.py: seconds_per_chunk = 2.0 * second_per_grid_ts[0]
            seconds_per_chunk=2.0 * second_per_grid[0] if second_per_grid else None,
        )

        model_inputs = self._to_device({**tokenized, **audio_inputs, **video_inputs})
        if audios:
            model_inputs["input_raw_wav"] = [w.to(self.device) for w in raw_wavs]
            model_inputs["use_audio_in_video"] = use_audio_in_video

        embeds = self.model(**model_inputs, pred_embeds=True, return_dict=True)
        return torch.nn.functional.normalize(embeds.mllm_embeds.float(), p=2, dim=-1)

    def _to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Move every tensor in ``inputs`` to the model's device, leaving the rest."""
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def _collate(self, inputs: list[dict[str, Any]]) -> BatchedInput:
        for row in inputs:
            if "video" in row:
                row["video_duration"] = row["video"].metadata.end_stream_seconds
        return self.collator(inputs)

    @torch.inference_mode()
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
            inputs = DataLoader(
                inputs.dataset,
                batch_size=self.video_batch_size,
                collate_fn=self._collate,
                num_workers=inputs.num_workers,
                shuffle=False,
            )
        elif "audio" in features:
            inputs.collate_fn = self._collate

        all_embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, desc="Encoding"):
            all_embeddings.append(self._encode_batch(batch).cpu())
        return torch.cat(all_embeddings, dim=0).float()


_OMNIRETRIEVER_CITATION = r"""
@article{liu2026omniretriever,
  title={OmniRetriever: Any-to-Any Audio-Video-Text Retrieval via Fusion-as-Teacher Distillation},
  author={Liu, Yunze},
  journal={arXiv preprint arXiv:2605.26641},
  year={2026}
}
"""

omniretriever_7b = ModelMeta(
    loader=OmniRetrieverWrapper,
    name="YunzeLiu/OmniRetriever-7B",
    revision="99328f1c5ce88695fa7070aac5b4a817aab60698",
    release_date="2026-05-27",
    languages=["eng-Latn"],
    n_parameters=9_417_532_287,
    n_embedding_parameters=544_997_376,
    memory_usage_mb=18703,
    max_tokens=32768,
    embed_dim=3584,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/yunzeliu/Omni-Retriever",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/YunzeLiu/OmniRetriever-7B",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    adapted_from="tsinghua-ee/WAVE-7B",
    superseded_by=None,
    modalities=["text", "audio", "video"],
    model_type=["dense"],
    citation=_OMNIRETRIEVER_CITATION,
    # "audio" is appended automatically from `modalities`; "video" (torchcodec)
    # is not, and "omniretriever" carries peft plus the transformers pin the
    # WAVE-7B remote code needs.
    extra_requirements_groups=["omniretriever", "video"],
)

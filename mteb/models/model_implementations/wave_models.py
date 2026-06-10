"""WAVE-7B: a Qwen2.5-Omni-Thinker based omni embedding model.

WAVE (https://github.com/TCL606/WAVE, https://huggingface.co/tsinghua-ee/WAVE-7B,
arXiv:2509.21990) produces "prompt-aware" embeddings over text, audio, silent video,
and audio-visual inputs in a shared space.

The HF checkpoint only ships weights + a vanilla ``Qwen2_5OmniThinkerForConditionalGeneration``
config, so the embedding path cannot be reproduced with stock ``transformers``. WAVE's real
modeling code (a BEATs dual audio encoder + hierarchical "all-layer" feature fusion) lives in
the upstream repo, vendored here as a git submodule at ``external/WAVE``. This wrapper imports
that code and replicates WAVE's ``--pred_embeds`` evaluation path:

    model(**inputs, pred_embeds=True) -> outputs.mllm_embeds

where ``mllm_embeds`` is the concatenation of the last-token hidden state of every transformer
layer (``classify_type="all_layer"``) passed through a learned ``classify_linear`` head.

Two external artifacts are required at load time and are NOT in this repo:
- the WAVE-7B weights (downloaded from the HF Hub), and
- the BEATs backbone checkpoint ``BEATs_iter3_plus.pt`` (Microsoft BEATs iter3+), pointed to by
  the ``beats_path`` loader kwarg or the ``WAVE_BEATS_PATH`` environment variable.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator, VideoCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

# Location of the vendored WAVE upstream code (git submodule).
_WAVE_REPO_PATH = Path(__file__).resolve().parents[3] / "external" / "WAVE"

# Default per-modality prompt used by WAVE's retrieval evaluation. WAVE is prompt-aware; these
# mirror the prompts in WAVE's eval configs (scripts/ret_*.json).
_DEFAULT_MEDIA_PROMPTS = {
    "audio": "Please describe the audio.",
    "video": "Please describe the video.",
    "image": "Please describe the image.",
}

# Audio sampling rate expected by WAVE (Whisper feature extractor + BEATs).
_SAMPLING_RATE = 16000


class Wave7BWrapper(AbsEncoder):
    """Faithful MTEB wrapper around WAVE-7B's ``--pred_embeds`` embedding path.

    One item is encoded at a time (matching WAVE's batch-size-1 evaluation). Each item is turned
    into WAVE's per-sample model inputs by reusing the upstream preprocessing
    (``process_audio``, ``process_omni_conversations``, ``replace_multimodal_special_tokens``),
    then ``model(**inputs, pred_embeds=True)`` is run and ``outputs.mllm_embeds`` is L2-normalized.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        *,
        beats_path: str | None = None,
        wave_repo_path: str | None = None,
        sim_temperature: float = 0.01,
        max_audio_length_seconds: int = 300,
        fps: float = 2.0,
        max_frames: int = 128,
        attn_implementation: str = "flash_attention_2",
        **kwargs: Any,
    ) -> None:
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.fps = fps
        self.max_frames = max_frames
        self.max_samples = int(max_audio_length_seconds * _SAMPLING_RATE)

        beats_path = beats_path or os.environ.get("WAVE_BEATS_PATH")
        if beats_path is None:
            raise ValueError(
                "WAVE-7B requires the BEATs backbone checkpoint 'BEATs_iter3_plus.pt'. "
                "Set the `beats_path` loader kwarg or the WAVE_BEATS_PATH environment variable. "
                "Download from the Microsoft unilm BEATs release."
            )
        if not Path(beats_path).is_file():
            raise FileNotFoundError(f"BEATs checkpoint not found at: {beats_path}")

        repo_path = Path(wave_repo_path) if wave_repo_path else _WAVE_REPO_PATH
        if not (repo_path / "qwenvl").is_dir():
            raise FileNotFoundError(
                f"WAVE upstream code not found at {repo_path}. Initialize the submodule: "
                "`git submodule update --init --recursive external/WAVE`."
            )
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        # Heavy / WAVE-specific imports are deferred so the registry can be built without
        # WAVE's dependencies installed.
        from qwenvl.data.data_qwen import LazySupervisedDataset
        from qwenvl.data.processing_qwen2_5_omni import Qwen2_5OmniProcessor
        from qwenvl.model.qwen2_5_omni.configuration_qwen2_5_omni import (
            Qwen2_5OmniThinkerConfig,
        )
        from qwenvl.model.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniThinkerForConditionalGeneration,
        )
        from qwenvl.train.argument import DataArguments

        self._apply_liger_kernel()

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_name, revision=revision
        )
        tokenizer = self.processor.tokenizer

        config = Qwen2_5OmniThinkerConfig.from_pretrained(model_name, revision=revision)
        config.train_classify = True
        config.classify_type = "all_layer"
        config.sim_temperature = sim_temperature
        config.audio_config.beats_path = beats_path
        config.audio_config.beats_only = False

        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_name,
            revision=revision,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        # The BEATs backbone weights are not in the WAVE-7B checkpoint; load them separately.
        beats_ckpt = torch.load(beats_path, map_location="cpu")
        self.model.beats.load_state_dict(beats_ckpt["model"])

        self.model.eval()
        self.model.to(self.device)

        # Reuse WAVE's preprocessing helpers. The dataset is constructed over an empty json so we
        # can call its methods (process_audio, process_omni_conversations) and reuse the image
        # processor configuration WAVE applies in __init__.
        data_args = DataArguments()
        self._empty_json = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        self._empty_json.write("[]")
        self._empty_json.flush()
        data_args.dataset_use = self._empty_json.name
        data_args.omni_processor = self.processor
        data_args.video_max_frames = max_frames
        data_args.video_min_frames = 1
        data_args.base_interval = 1.0 / fps
        data_args.use_beats = True
        data_args.beats_only = False
        data_args.train_classify = True
        data_args.run_test = False
        self._ds = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)

    @staticmethod
    def _apply_liger_kernel() -> None:
        """Patch WAVE's modeling module with Liger kernels, as WAVE's eval entrypoint does."""
        try:
            from liger_kernel.transformers.qwen2vl_mrope import (
                liger_multimodal_rotary_pos_emb,
            )
            from liger_kernel.transformers.rms_norm import LigerRMSNorm
            from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
            from qwenvl.model.qwen2_5_omni import modeling_qwen2_5_omni

            modeling_qwen2_5_omni.apply_multimodal_rotary_pos_emb = (
                liger_multimodal_rotary_pos_emb
            )
            modeling_qwen2_5_omni.Qwen2RMSNorm = LigerRMSNorm
            modeling_qwen2_5_omni.Qwen2MLP = LigerSwiGLUMLP
        except Exception as e:
            logger.warning("Could not apply Liger kernels for WAVE: %s", e)

    def _process_image(self, image: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess a PIL image, mirroring WAVE's ``process_image_unified``."""
        from PIL import ImageOps

        processor = copy.deepcopy(self.processor.image_processor)
        processor.max_pixels = self._ds.data_args.image_max_frame_pixels
        processor.min_pixels = self._ds.data_args.image_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels

        image = image.convert("RGB")
        width, height = image.size
        if width < 28 or height < 28:
            pad_width = max(0, 28 - width)
            pad_height = max(0, 28 - height)
            left, top = pad_width // 2, pad_height // 2
            image = ImageOps.expand(
                image,
                border=(left, top, pad_width - left, pad_height - top),
                fill=(0, 0, 0),
            )
        visual = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        grid_thw = visual["image_grid_thw"][0]
        return image_tensor, grid_thw

    def _process_video(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        """Preprocess pre-sampled video frames, mirroring WAVE's ``video_decord`` tail.

        MTEB's ``VideoCollator`` already samples frames at ``self.fps``. WAVE's decord path yields
        ``(F, H, W, C)`` uint8; torchcodec yields ``(F, C, H, W)``, so permute when needed.
        """
        if frames.ndim == 4 and frames.shape[1] == 3 and frames.shape[-1] != 3:
            frames = frames.permute(0, 2, 3, 1)
        frames = frames.contiguous()
        image_processor = self.processor.image_processor
        video_proc = image_processor(images=None, videos=frames, return_tensors="pt")
        video_second_per_grid = [image_processor.temporal_patch_size / self.fps]
        return (
            video_proc["pixel_values_videos"],
            video_proc["video_grid_thw"],
            video_second_per_grid,
        )

    def _build_inputs(  # noqa: PLR0914
        self,
        *,
        text: str | None,
        image: Any,
        audio: Any,
        video: Any,
        instruction: str | None,
    ) -> dict[str, Any]:
        """Build WAVE's per-sample model inputs, mirroring ``LazySupervisedDataset._get_item``."""
        image_tensor = grid_thw = None
        video_tensor = video_grid_thw = second_per_grid_ts = None
        audio_inputs = audio_lengths = raw_wav = None

        if image is not None:
            image_tensor, grid_thw = self._process_image(image)
            grid_thw = [grid_thw.unsqueeze(0)]
        if video is not None:
            video_tensor, video_grid_thw, second_per_grid_ts = self._process_video(
                video
            )
            video_grid_thw = [video_grid_thw]
        if audio is not None:
            arr = audio["array"] if isinstance(audio, dict) else audio
            arr = np.asarray(arr, dtype=np.float32)
            audio_inputs, audio_lengths, raw_wav = self._ds.process_audio(audio_wav=arr)
            if raw_wav is not None and len(raw_wav[0]) < _SAMPLING_RATE:
                sil = np.zeros(_SAMPLING_RATE - len(raw_wav[0]), dtype=raw_wav[0].dtype)
                raw_wav[0] = np.concatenate((raw_wav[0], sil), axis=0)

        # Pick the media tag (priority video > audio > image, matching WAVE's data handling).
        if video is not None:
            media = "video"
        elif audio is not None:
            media = "audio"
        elif image is not None:
            media = "image"
        else:
            media = None

        if media is not None:
            prompt = instruction or _DEFAULT_MEDIA_PROMPTS[media]
            value = f"<{media}>\n{prompt}"
        else:
            value = text if text is not None else ""
            if instruction:
                value = f"{instruction}\n{value}"

        conversations = [
            {"from": "human", "value": value},
            {"from": "gpt", "value": ""},
        ]
        txt, _ = self._ds.process_omni_conversations(conversations, "retrieval")
        txt = self.processor.replace_multimodal_special_tokens(
            txt,
            iter(audio_lengths[0]) if audio_inputs is not None else iter([]),
            iter(grid_thw[0]) if grid_thw is not None else iter([]),
            iter(video_grid_thw[0]) if video_grid_thw is not None else iter([]),
            video_second_per_grid=iter(second_per_grid_ts)
            if second_per_grid_ts is not None
            else iter([]),
            use_audio_in_video=False,
            position_id_per_seconds=25,
            seconds_per_chunk=None,
        )
        # BEATs interleaves a second audio token per frame; double the audio placeholders.
        if audio_inputs is not None:
            txt[0] = txt[0].replace("<|AUDIO|>", "<|AUDIO|><|AUDIO|>")

        token_res = self.processor.tokenizer(
            txt, padding=True, padding_side="left", return_tensors="pt"
        )
        return {
            "input_ids": token_res["input_ids"],
            "attention_mask": token_res["attention_mask"],
            "pixel_values": image_tensor,
            "image_grid_thw": grid_thw[0] if grid_thw is not None else None,
            "pixel_values_videos": video_tensor,
            "video_grid_thw": video_grid_thw[0] if video_grid_thw is not None else None,
            "video_second_per_grid": second_per_grid_ts[0]
            if second_per_grid_ts is not None
            else None,
            "input_features": audio_inputs[0]["input_features"]
            if audio_inputs is not None
            else None,
            "feature_attention_mask": audio_inputs[0]["feature_attention_mask"]
            if audio_inputs is not None
            else None,
            "input_raw_wav": torch.from_numpy(raw_wav[0])
            if raw_wav is not None
            else None,
            "type": "retrieval",
        }

    def _to_model_kwargs(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        """Move a per-sample dict to device, mirroring WAVE's eval loop in train_qwen.py."""
        raw_wav = data_dict.pop("input_raw_wav", None)
        kwargs: dict[str, Any] = {}
        for k, v in data_dict.items():
            if v is None:
                continue
            if k == "video_second_per_grid":
                kwargs[k] = torch.tensor([v], device=self.device)
            elif isinstance(v, torch.Tensor):
                kwargs[k] = v.to(self.device)
            else:
                kwargs[k] = v
        if raw_wav is not None:
            kwargs["input_raw_wav"] = [raw_wav.to(self.device)]
        kwargs["types"] = [kwargs.pop("type", "retrieval")]
        kwargs["pred_embeds"] = True
        return kwargs

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
        has_video = "video" in features
        has_audio = "audio" in features
        if has_video:
            inputs.collate_fn = VideoCollator(
                target_sampling_rate=_SAMPLING_RATE,
                fps=self.fps,
                max_frames=self.max_frames,
                max_samples=self.max_samples,
            )
        elif has_audio:
            inputs.collate_fn = AudioCollator(
                target_sampling_rate=_SAMPLING_RATE,
                max_samples=self.max_samples,
            )

        # Use an explicitly-defined task prompt if present; otherwise fall back to WAVE's
        # per-modality default prompts (set in `_build_inputs`).
        instruction: str | None = None
        prompt = task_metadata.prompt
        if isinstance(prompt, dict) and prompt_type is not None:
            instruction = prompt.get(prompt_type.value)
        elif isinstance(prompt, str) and prompt:
            instruction = prompt

        all_embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, desc="Encoding"):
            texts = batch.get("text", [])
            images = batch.get("image", [])
            audios = batch.get("audio", [])
            videos = batch.get("video", [])
            batch_size = max(len(texts), len(images), len(audios), len(videos))
            for i in range(batch_size):
                data_dict = self._build_inputs(
                    text=texts[i] if i < len(texts) else None,
                    image=images[i] if i < len(images) else None,
                    audio=audios[i] if i < len(audios) else None,
                    video=videos[i] if i < len(videos) else None,
                    instruction=instruction,
                )
                model_kwargs = self._to_model_kwargs(data_dict)
                outputs = self.model(**model_kwargs)
                emb = torch.nn.functional.normalize(
                    outputs.mllm_embeds.float(), p=2, dim=-1
                )
                all_embeddings.append(emb.cpu())
        return torch.cat(all_embeddings, dim=0)


WAVE_CITATION = """@article{cheng2025wave,
    title={WAVE: Learning Unified & Versatile Audio-Visual Embeddings with Multimodal LLM},
    author={Cheng, Changli and others},
    year={2025},
    eprint={2509.21990},
    archivePrefix={arXiv},
    primaryClass={cs.MM},
    url={https://arxiv.org/abs/2509.21990},
}"""

wave_7b = ModelMeta(
    loader=Wave7BWrapper,
    name="tsinghua-ee/WAVE-7B",
    revision="6d42651d34bf1a7d83d5779397d6ce0316a4cf4f",
    release_date="2026-01-28",
    languages=["eng-Latn"],
    n_parameters=7_000_000_000,
    memory_usage_mb=None,
    max_tokens=32768,
    embed_dim=3584,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/TCL606/WAVE",
    public_training_data=None,
    framework=["PyTorch", "Transformers"],
    reference="https://huggingface.co/tsinghua-ee/WAVE-7B",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=None,
    adapted_from="Qwen/Qwen2.5-Omni-7B",
    superseded_by=None,
    modalities=["text", "audio", "video"],
    model_type=["dense"],
    extra_requirements_groups=["wave"],
    citation=WAVE_CITATION,
)

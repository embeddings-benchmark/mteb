from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import VideoCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

_SAMPLING_RATE = 16_000

# Per-modality prompt used when a task provides no instruction, mirroring the
# prompts in WAVE's own eval configs (scripts/ret_*.json in the WAVE repo).
_DEFAULT_MEDIA_PROMPTS = {
    "audio": "Please describe the audio.",
    "video": "Please describe the video.",
    "image": "Please describe the image.",
}


class _DurationVideoCollator(VideoCollator):
    """VideoCollator that also records each video's duration.

    WAVE derives its per-video frame rate as sampled_frames / video_length
    (see ``_process_video``), so the source duration has to be captured
    before the parent collator overwrites ``row["video"]`` with the
    already-frame-sampled tensor.
    """

    def __call__(self, inputs: list[dict[str, Any]]) -> BatchedInput:
        for row in inputs:
            video = row.get("video")
            metadata = getattr(video, "metadata", None)
            row["video_duration"] = getattr(metadata, "end_stream_seconds", None)
        return super().__call__(inputs)


class Wave7BWrapper(AbsEncoder):
    """Wrapper around WAVE-7B's ``pred_embeds`` embedding path.

    WAVE (https://huggingface.co/tsinghua-ee/WAVE-7B, arXiv:2509.21990) is a
    Qwen2.5-Omni-Thinker based model producing prompt-aware embeddings for
    text, audio, silent video, and synchronized audio-visual inputs, via a
    dual audio encoder (Whisper + BEATs) and an "all-layer" feature-fusion
    head, rather than plain last-hidden-state pooling.

    The checkpoint ships its full modeling/processor code (auto_map in its
    config.json), so it loads directly via ``trust_remote_code`` -- no
    vendored copy of the upstream training repo is needed. What upstream
    doesn't ship is the exact input-construction logic used at training/eval
    time (WAVE's ``process_audio`` / ``process_omni_conversations``, from
    ``qwenvl/data/data_qwen.py`` in the WAVE GitHub repo); that logic is
    reimplemented here directly, matched against WAVE's own eval entrypoint
    (``qwenvl/train/train_qwen.py``, invoked by ``scripts/direct_test.sh``):

    - Encoding runs one item at a time (WAVE's own eval batch size is 1).
    - Media items go through ``model(**inputs, pred_embeds=True).mllm_embeds``:
      the last-token hidden state of every transformer layer, concatenated
      and passed through the model's ``classify_linear`` head
      (``classify_type="all_layer"``).
    - Text-only items use WAVE's separate "label" path instead: no chat
      template, last token of the final layer only, no ``classify_linear``.
    - The checkpoint's BEATs submodule is deliberately reloaded from the
      standalone ``BEATs_iter3_plus_AS2M.pt`` file (also shipped in the same
      repo) after the main ``from_pretrained`` call -- WAVE's own eval script
      does this unconditionally whenever BEATs is enabled, so it's replicated
      here to reproduce the paper's reported numbers exactly.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        *,
        max_audio_length_seconds: float | None = 300.0,
        fps: float = 2.0,
        max_frames: int = 128,
        attn_implementation: str = "sdpa",
        **kwargs: Any,
    ) -> None:
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModel
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.fps = fps
        self.max_frames = max_frames
        self.max_samples = (
            int(max_audio_length_seconds * _SAMPLING_RATE)
            if max_audio_length_seconds is not None
            else None
        )

        # AutoProcessor can't resolve WAVE's custom Qwen2_5OmniProcessor since
        # the repo has no standalone processor_config.json with an auto_map
        # entry (only config.json's auto_map declares it) -- it silently
        # falls back to a plain tokenizer instead. Load the class directly.
        processor_cls = get_class_from_dynamic_module(
            "processing_qwen2_5_omni.Qwen2_5OmniProcessor",
            model_name,
            revision=revision,
        )
        self.processor = processor_cls.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )

        beats_path = hf_hub_download(
            repo_id=model_name,
            filename="BEATs_iter3_plus_AS2M.pt",
            revision=revision,
        )
        # train_classify/classify_type/sim_temperature are already baked into
        # the saved config.json (train_classify=True, classify_type="all_layer");
        # only beats_path needs overriding to point at the locally downloaded file.
        config = AutoConfig.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )
        config.audio_config.beats_path = beats_path
        config.audio_config.beats_only = False

        # device_map loads shards directly onto the target device instead of
        # fully materializing the model on CPU and copying it over afterwards
        # -- meaningfully lower peak memory for a 7B model.
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map=self.device,
        )
        # WAVE's own eval entrypoint reloads the BEATs submodule from the
        # standalone checkpoint after `from_pretrained`, regardless of what
        # `from_pretrained` already populated it with (see train_qwen.py's
        # `run_test` path). Replicated here for a faithful reproduction.
        # load_state_dict copies onto whatever device the target params are
        # already on, so this works fine with the model already placed.
        beats_ckpt = torch.load(beats_path, map_location="cpu")
        self.model.beats.load_state_dict(beats_ckpt["model"])

        self.model.eval()

    def _process_image(self, image: Any) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import ImageOps

        processor = copy.deepcopy(self.processor.image_processor)
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
        return image_tensor, visual["image_grid_thw"][0]

    def _process_video(
        self, frames: torch.Tensor, duration: float | None
    ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        """Mirror WAVE's video_decord tail on already-sampled frames.

        MTEB's collator samples frames at self.fps and yields (F, C, H, W)
        (torchcodec); WAVE's own decord path yields (F, H, W, C), which is
        what the bundled image processor expects.

        WAVE computes fps = sampled_frames / video_length, which can differ
        from the nominal rate whenever the frame cap binds or a short video
        keeps all its frames, and derives video_second_per_grid from that
        actual fps. Replicated here using the sampled frame count and the
        source video's duration.
        """
        if frames.ndim == 4 and frames.shape[1] == 3 and frames.shape[-1] != 3:
            frames = frames.permute(0, 2, 3, 1)
        frames = frames.contiguous()
        image_processor = self.processor.image_processor
        video_proc = image_processor(images=None, videos=frames, return_tensors="pt")
        fps = frames.shape[0] / duration if duration else self.fps
        video_second_per_grid = [image_processor.temporal_patch_size / fps]
        return (
            video_proc["pixel_values_videos"],
            video_proc["video_grid_thw"],
            video_second_per_grid,
        )

    def _process_audio(
        self, array: np.ndarray
    ) -> tuple[dict[str, torch.Tensor], int, np.ndarray]:
        """Mirror WAVE's process_audio: chunk into 300s segments for BEATs.

        WAVE never truncates audio itself; long clips are chunked into 300s
        segments and each segment is fed through the feature extractor
        separately, then concatenated. `max_audio_length_seconds` is an
        opt-in cap applied before this (for OOM safety on real-world data),
        not part of WAVE's own preprocessing.
        """
        array = np.asarray(array, dtype=np.float32)
        feature_extractor = self.processor.feature_extractor
        segment_samples = 300 * _SAMPLING_RATE
        segments = [
            array[k : k + segment_samples]
            for k in range(0, len(array), segment_samples)
        ]

        feature_attention_masks = []
        input_features = []
        audio_length = 0
        for segment in segments:
            if segment.shape[0] < _SAMPLING_RATE:
                segment = np.pad(  # noqa: PLW2901
                    segment, (0, _SAMPLING_RATE - segment.shape[0])
                )
            features = feature_extractor(
                segment,
                sampling_rate=_SAMPLING_RATE,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
            attn = features["attention_mask"]
            feature_attention_masks.append(attn)
            input_features.append(features["input_features"])
            segment_length = (attn.sum(-1) - 1) // 2 + 1
            audio_length += (segment_length - 2) // 2 + 1

        raw_wav = array
        if len(raw_wav) < _SAMPLING_RATE:
            raw_wav = np.pad(raw_wav, (0, _SAMPLING_RATE - len(raw_wav)))

        return (
            {
                "feature_attention_mask": torch.cat(feature_attention_masks, dim=0),
                "input_features": torch.cat(input_features, dim=0),
            },
            int(audio_length.item()),
            raw_wav,
        )

    def _build_inputs(
        self,
        *,
        text: str | None,
        image: Any,
        audio: Any,
        video: Any,
        instruction: str | None,
        video_duration: float | None,
    ) -> dict[str, Any]:
        """Build one sample's model inputs, mirroring LazySupervisedDataset._get_item."""
        image_tensor = grid_thw = None
        video_tensor = video_grid_thw = second_per_grid_ts = None
        audio_inputs = audio_length = raw_wav = None

        if image is not None:
            image_tensor, grid_thw = self._process_image(image)
        if video is not None:
            video_tensor, video_grid_thw, second_per_grid_ts = self._process_video(
                video, video_duration
            )
        if audio is not None:
            array = audio["array"] if isinstance(audio, dict) else audio
            audio_inputs, audio_length, raw_wav = self._process_audio(array)

        media = (
            "video" if video is not None else "audio" if audio is not None else "image"
        )
        content = text or instruction or _DEFAULT_MEDIA_PROMPTS[media]
        conversation = [
            {
                "role": "user",
                "content": [{"type": media}, {"type": "text", "text": content}],
            }
        ]
        text_input = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False, tokenize=False
        )
        text_input = text_input[0].split("<|im_start|>user\n")[-1].strip()

        text_input = self.processor.replace_multimodal_special_tokens(
            [text_input],
            iter([audio_length]) if audio_inputs is not None else iter([]),
            iter([grid_thw]) if grid_thw is not None else iter([]),
            iter([video_grid_thw[0]]) if video_grid_thw is not None else iter([]),
            video_second_per_grid=iter(second_per_grid_ts)
            if second_per_grid_ts is not None
            else iter([]),
            use_audio_in_video=audio_inputs is not None and video is not None,
            position_id_per_seconds=25,
            seconds_per_chunk=2.0 * second_per_grid_ts[0]
            if second_per_grid_ts is not None
            else None,
        )
        # BEATs interleaves a second audio token per frame on top of the
        # Whisper tower's placeholders.
        if audio_inputs is not None:
            text_input[0] = text_input[0].replace("<|AUDIO|>", "<|AUDIO|><|AUDIO|>")

        tokenized = self.processor.tokenizer(
            text_input, padding=True, padding_side="left", return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "pixel_values": image_tensor,
            "image_grid_thw": grid_thw.unsqueeze(0) if grid_thw is not None else None,
            "pixel_values_videos": video_tensor,
            "video_grid_thw": video_grid_thw,
            "video_second_per_grid": second_per_grid_ts[0]
            if second_per_grid_ts is not None
            else None,
            "input_features": audio_inputs["input_features"]
            if audio_inputs is not None
            else None,
            "feature_attention_mask": audio_inputs["feature_attention_mask"]
            if audio_inputs is not None
            else None,
            "input_raw_wav": [torch.from_numpy(raw_wav)]
            if raw_wav is not None
            else None,
            "use_audio_in_video": video_grid_thw is not None
            and audio_inputs is not None,
        }

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        """Embed text via WAVE's "label" path: bare text, no chat template,
        no instruction, last token of the final layer, no classify_linear
        head (modeling_qwen2_5_omni.py's `label_ids is not None` branch).
        """
        if not text.endswith("<|im_end|>"):
            text += "<|im_end|>"
        tokenized = self.processor.tokenizer(
            [text], padding=True, padding_side="left", return_tensors="pt"
        )
        inputs_embeds = self.model.get_input_embeddings()(
            tokenized["input_ids"].to(self.device)
        )
        outputs = self.model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=tokenized["attention_mask"].to(self.device),
            return_dict=True,
        )
        return outputs[0][:, -1, :]

    def _to_model_kwargs(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        raw_wav = data_dict.pop("input_raw_wav", None)
        kwargs: dict[str, Any] = {}
        for key, value in data_dict.items():
            if value is None:
                continue
            if key == "video_second_per_grid":
                kwargs[key] = torch.tensor([value], device=self.device)
            elif isinstance(value, torch.Tensor):
                kwargs[key] = value.to(self.device)
            else:
                kwargs[key] = value
        if raw_wav is not None:
            kwargs["input_raw_wav"] = [w.to(self.device) for w in raw_wav]
        kwargs["pred_embeds"] = True
        return kwargs

    @torch.no_grad()
    def encode(  # noqa: PLR0914
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        features = inputs.dataset.features
        if "video" in features:
            inputs.collate_fn = _DurationVideoCollator(
                target_sampling_rate=_SAMPLING_RATE,
                fps=self.fps,
                max_frames=self.max_frames,
                max_samples=self.max_samples,
            )
        elif "audio" in features:
            from mteb.models.modality_collators import AudioCollator

            inputs.collate_fn = AudioCollator(
                target_sampling_rate=_SAMPLING_RATE, max_samples=self.max_samples
            )

        instruction: str | None = None
        prompt = task_metadata.prompt
        if isinstance(prompt, dict) and prompt_type is not None:
            instruction = prompt.get(prompt_type.value)
        elif isinstance(prompt, str) and prompt:
            instruction = prompt

        all_embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding"):
            texts = batch.get("text", [])
            images = batch.get("image", [])
            audios = batch.get("audio", [])
            videos = batch.get("video", [])
            durations = batch.get("video_duration", [])
            batch_size = max(len(texts), len(images), len(audios), len(videos))
            for i in range(batch_size):
                text_i = texts[i] if i < len(texts) else None
                image_i = images[i] if i < len(images) else None
                audio_i = audios[i] if i < len(audios) else None
                video_i = videos[i] if i < len(videos) else None
                if image_i is None and audio_i is None and video_i is None:
                    raw = self._encode_text(text_i or "")
                else:
                    data_dict = self._build_inputs(
                        text=text_i,
                        image=image_i,
                        audio=audio_i,
                        video=video_i,
                        instruction=instruction,
                        video_duration=durations[i] if i < len(durations) else None,
                    )
                    model_kwargs = self._to_model_kwargs(data_dict)
                    raw = self.model(**model_kwargs).mllm_embeds
                embedding = torch.nn.functional.normalize(raw.float(), p=2, dim=-1)
                all_embeddings.append(embedding.cpu())
        return torch.cat(all_embeddings, dim=0).numpy()


WAVE_CITATION = """@inproceedings{tang2026wave,
    title={{WAVE}: Learning Unified \\& Versatile Audio-Visual Embeddings with Multimodal {LLM}},
    author={Changli Tang and Qinfan Xiao and Ke Mei and Tianyi Wang and Fengyun Rao and Chao Zhang},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=MiV3WXDYJb},
}"""

wave_7b = ModelMeta(
    loader=Wave7BWrapper,
    name="tsinghua-ee/WAVE-7B",
    languages=["eng-Latn"],
    open_weights=True,
    revision="7d51cdaecfaabb9c529a447249cd4c2a6df8ce5b",
    release_date="2026-02-11",
    max_tokens=131_072,
    n_parameters=9_410_651_007,
    n_embedding_parameters=0,
    memory_usage_mb=17949,
    embed_dim=3584,
    license="apache-2.0",
    reference="https://huggingface.co/tsinghua-ee/WAVE-7B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=True,
    public_training_code="https://github.com/TCL606/WAVE",
    public_training_data=None,
    training_datasets=None,
    adapted_from="Qwen/Qwen2.5-Omni-7B",
    modalities=["text", "image", "audio", "video"],
    model_type=["dense"],
    citation=WAVE_CITATION,
    extra_requirements_groups=["wave"],
)

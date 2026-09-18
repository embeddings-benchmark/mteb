from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType


BIOVITA_MODEL_ID = "risashinoda/BioVITA"
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"

TARGET_SAMPLING_RATE = 48_000
AUDIO_SECONDS = 10.0
AUDIO_SAMPLES = int(TARGET_SAMPLING_RATE * AUDIO_SECONDS)


def _prepare_biovita_audio(
    audio: dict[str, Any],
    target_sr: int = TARGET_SAMPLING_RATE,
    seconds: float = AUDIO_SECONDS,
    thr_ratio: float = 0.01,
    preroll: float = 0.2,
) -> torch.Tensor:
    """Reproduce BioVITA's official onset-based audio preprocessing."""
    from torchaudio.functional import resample

    n_samples = int(target_sr * seconds)

    waveform = torch.as_tensor(audio["array"], dtype=torch.float32)
    sampling_rate = int(audio["sampling_rate"])

    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    elif waveform.ndim != 1:
        raise ValueError(f"Expected 1D or 2D audio, got shape={tuple(waveform.shape)}")

    if sampling_rate != target_sr and waveform.numel() > 1:
        try:
            waveform = resample(waveform, sampling_rate, target_sr)
        except Exception:
            return torch.zeros(n_samples, dtype=torch.float32)

    if waveform.numel() <= 1:
        return torch.zeros(n_samples, dtype=torch.float32)

    energy = waveform.abs()
    threshold = float(energy.max().item()) * thr_ratio
    nonzero = (energy > threshold).nonzero(as_tuple=True)[0]

    start = (
        0
        if len(nonzero) == 0
        else max(
            0,
            int(nonzero[0].item()) - int(preroll * target_sr),
        )
    )

    waveform = waveform[start : start + n_samples]

    if waveform.numel() < n_samples:
        waveform = F.pad(waveform, (0, n_samples - waveform.numel()))

    return waveform[:n_samples].to(torch.float32)


class _BioVITAAudioCollator:
    """Prepare HF Audio inputs exactly as the BioVITA reference implementation."""

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
        collated = []

        for row in rows:
            processed_row = dict(row)
            waveform = _prepare_biovita_audio(processed_row["audio"])

            processed_row["audio"] = {
                "array": waveform.numpy(),
                "sampling_rate": TARGET_SAMPLING_RATE,
            }
            collated.append(processed_row)

        return {key: [row[key] for row in collated] for key in collated[0]}


def _load_biovita_text_image_model(
    model_name: str,
    revision: str | None,
    device: torch.device,
) -> tuple[Any, Any, Any]:
    """Load BioVITA's OpenCLIP components from a pinned HF revision."""
    import json

    import open_clip
    from huggingface_hub import hf_hub_download

    clip_config_path = hf_hub_download(
        repo_id=model_name,
        filename="open_clip_config.json",
        revision=revision,
    )
    clip_weights_path = hf_hub_download(
        repo_id=model_name,
        filename="open_clip_pytorch_model.bin",
        revision=revision,
    )

    with Path(clip_config_path).open(encoding="utf-8") as config_file:
        clip_config = json.load(config_file)

    model_cfg = clip_config["model_cfg"]
    expected_cfg = open_clip.get_model_config("ViT-L-14")
    if model_cfg != expected_cfg:
        raise RuntimeError(
            "BioVITA OpenCLIP config does not match the expected ViT-L-14 architecture."
        )

    preprocess_cfg = clip_config["preprocess_cfg"]

    model, image_preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained=clip_weights_path,
        image_mean=tuple(preprocess_cfg["mean"]),
        image_std=tuple(preprocess_cfg["std"]),
        image_interpolation=preprocess_cfg["interpolation"],
        image_resize_mode=preprocess_cfg["resize_mode"],
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    model.eval().to(device)
    return model, image_preprocess, tokenizer


class BioVITAWrapper(AbsEncoder):
    """MTEB wrapper for BioVITA audio-image-text embeddings."""

    def __init__(
        self,
        model_name: str = BIOVITA_MODEL_ID,
        revision: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ):
        from huggingface_hub import hf_hub_download
        from transformers import ClapModel, ClapProcessor

        self.model_name = model_name
        self.revision = revision
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        (
            self.txt_img_model,
            self.image_preprocess,
            self.tokenizer,
        ) = _load_biovita_text_image_model(
            model_name,
            revision,
            self.device,
        )

        # Match the official BioVITA CLAP backbone.
        self.audio_processor = ClapProcessor.from_pretrained(
            CLAP_MODEL_ID,
            revision=CLAP_REVISION,
        )

        clap_dtype = torch.float16 if self.device.type == "cuda" else None

        self.clap = ClapModel.from_pretrained(
            CLAP_MODEL_ID,
            revision=CLAP_REVISION,
            low_cpu_mem_usage=True,
            torch_dtype=clap_dtype,
        ).to(self.device)
        self.clap.eval()

        feature_extractor = getattr(self.audio_processor, "feature_extractor", None)
        if feature_extractor is not None:
            if hasattr(feature_extractor, "do_resample"):
                feature_extractor.do_resample = False
            if hasattr(feature_extractor, "return_attention_mask"):
                feature_extractor.return_attention_mask = False

        # Load BioVITA's fine-tuned CLAP weights and 512 -> 768 adapter.
        clap_path = hf_hub_download(
            repo_id=model_name,
            filename="clap_weights.pth",
            revision=revision,
        )
        state_dict = torch.load(
            clap_path,
            map_location="cpu",
            weights_only=True,
        )

        clap_state = state_dict.get("clap_audio", {})
        load_result = self.clap.load_state_dict(
            clap_state,
            strict=False,
        )

        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                "BioVITA CLAP checkpoint did not load cleanly. "
                f"Missing={load_result.missing_keys}, "
                f"unexpected={load_result.unexpected_keys}"
            )

        adapter_state = state_dict.get("audio_adapter", {})
        if "weight" not in adapter_state:
            raise RuntimeError(
                "BioVITA checkpoint does not contain audio_adapter.weight"
            )

        out_dim, in_dim = adapter_state["weight"].shape
        self.audio_adapter = torch.nn.Linear(
            in_dim,
            out_dim,
            bias=False,
        ).to(self.device)
        self.audio_adapter.load_state_dict(
            adapter_state,
            strict=True,
        )
        self.audio_adapter.eval()

        self.embed_dim = out_dim
        self._use_amp = self.device.type == "cuda"
        self._amp_dtype = torch.float16

        for parameter in self.txt_img_model.parameters():
            parameter.requires_grad = False
        for parameter in self.clap.parameters():
            parameter.requires_grad = False
        for parameter in self.audio_adapter.parameters():
            parameter.requires_grad = False

    @torch.inference_mode()
    def get_text_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
            desc="Encoding BioVITA text",
        ):
            tokens = self.tokenizer(list(batch["text"])).to(self.device)

            with torch.autocast(
                device_type="cuda",
                dtype=self._amp_dtype,
                enabled=self._use_amp,
            ):
                features = self.txt_img_model.encode_text(tokens)

            features = F.normalize(
                features,
                dim=-1,
            ).float()

            embeddings.append(features.cpu().numpy())

        return np.vstack(embeddings)

    @torch.inference_mode()
    def get_image_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
            desc="Encoding BioVITA images",
        ):
            image_tensor = torch.stack(
                [
                    self.image_preprocess(image.convert("RGB"))
                    for image in batch["image"]
                ]
            ).to(self.device)

            with torch.autocast(
                device_type="cuda",
                dtype=self._amp_dtype,
                enabled=self._use_amp,
            ):
                features = self.txt_img_model.encode_image(image_tensor)

            features = F.normalize(
                features,
                dim=-1,
            ).float()

            embeddings.append(features.cpu().numpy())

        return np.vstack(embeddings)

    @torch.inference_mode()
    def get_audio_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        inputs.collate_fn = _BioVITAAudioCollator()

        embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
            desc="Encoding BioVITA audio",
        ):
            waveforms = [audio["array"] for audio in batch["audio"]]

            audio_inputs = self.audio_processor(
                audio=waveforms,
                sampling_rate=TARGET_SAMPLING_RATE,
                return_tensors="pt",
                padding=True,
            )

            audio_inputs = {
                key: value.to(self.device) for key, value in audio_inputs.items()
            }

            model_dtype = next(self.clap.parameters()).dtype

            for key, value in audio_inputs.items():
                if torch.is_floating_point(value):
                    audio_inputs[key] = value.to(model_dtype)

            with torch.autocast(
                device_type="cuda",
                dtype=self._amp_dtype,
                enabled=self._use_amp,
            ):
                audio_outputs = self.clap.get_audio_features(**audio_inputs)

            features = self.audio_adapter(audio_outputs.pooler_output)
            features = F.normalize(
                features,
                dim=-1,
            ).float()

            embeddings.append(features.cpu().numpy())

        return np.vstack(embeddings)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        features = inputs.dataset.features

        modalities = [
            modality for modality in ("text", "image", "audio") if modality in features
        ]

        if not modalities:
            raise ValueError(
                f"No supported BioVITA modality found in features: {list(features)}"
            )

        embeddings = None

        for modality in modalities:
            if modality == "text":
                modality_embeddings = self.get_text_embeddings(
                    inputs,
                    **kwargs,
                )
            elif modality == "image":
                modality_embeddings = self.get_image_embeddings(
                    inputs,
                    **kwargs,
                )
            elif modality == "audio":
                modality_embeddings = self.get_audio_embeddings(
                    inputs,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unsupported BioVITA modality: {modality}")

            if embeddings is None:
                embeddings = modality_embeddings
            else:
                if len(embeddings) != len(modality_embeddings):
                    raise ValueError(
                        "All modalities must contain the same number of samples "
                        "for fused BioVITA embeddings."
                    )
                embeddings += modality_embeddings

        if embeddings is None:
            raise ValueError("BioVITA produced no embeddings.")

        if len(modalities) > 1:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings /= np.clip(
                norms,
                a_min=1e-12,
                a_max=None,
            )

        return embeddings


BIOVITA_CITATION = """
@inproceedings{shinoda2026biovita,
  title     = {BioVITA: Biological Dataset, Model, and Benchmark for Visual-Textual-Acoustic Alignment},
  author    = {Risa Shinoda and Kaede Shiohara and Nakamasa Inoue and Kuniaki Saito and Hiroaki Santo and Fumio Okura},
  booktitle = {CVPR},
  year      = {2026}
}
"""

biovita = ModelMeta(
    loader=BioVITAWrapper,
    name="risashinoda/BioVITA",
    revision="64aea8edfcc846842a167944f4a5a052d932d0b4",
    release_date="2026-05-13",
    languages=["eng-Latn"],
    modalities=["audio", "image", "text"],
    n_parameters=581_502_619,
    n_embedding_parameters=37_945_344,
    memory_usage_mb=2218,
    max_tokens=77,
    embed_dim=768,
    license="mit",
    open_weights=True,
    reference="https://huggingface.co/risashinoda/BioVITA",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers"],
    use_instructions=False,
    public_training_code="https://github.com/dahlian00/BioVITA",
    public_training_data="https://huggingface.co/datasets/risashinoda/BioVITA",
    training_datasets=None,
    model_type=["dense"],
    citation=BIOVITA_CITATION,
    extra_requirements_groups=[
        "image",
        "audio",
        "open_clip_torch",
        "transformers-v5",
    ],
)

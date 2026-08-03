"""InternVideo2-CLIP video and text encoders.

InternVideo2-CLIP pairs a 1B InternVideo2 video tower with the InternVL-C text
tower. The released checkpoint contains only the CLIP-stage deltas, so the loader
assembles the model from three separate Hugging Face repositories.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import FramesCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

_INSTALL_HINT = """InternVideo2-CLIP requires the upstream OpenGVLab package.

Install it from a local clone rather than a git URL: upstream's .gitmodules is missing
an entry for InternVideo3/InternVideo3_eval/lmms-eval, so pip's automatic submodule
init fails.

    git clone --depth 1 https://github.com/OpenGVLab/InternVideo.git
    pip install --no-deps -e InternVideo/InternVideo2/multi_modality
    pip install "mteb[internvideo2]"

--no-deps is deliberate: the upstream project pins apex, deepspeed and flash-attn,
none of which this inference path needs.
"""

# The HF repo for this model holds only the CLIP-stage deltas (47 tensors, ~7M
# params: the vision clip_projector, the learned temperature, rotary buffers).
# Both towers come from separate checkpoints resolved below.
_STAGE2_REPO = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
_INTERNVL_REPO = "OpenGVLab/InternVL"
_INTERNVL_TEXT_CKPT = "internvl_c_13b_224px.pth"

# LlamaConfig for the InternVL-C text tower. Inlined rather than downloaded so
# that loading needs one fewer remote artifact; mirrors
# OpenGVLab/InternVL:clip_benchmark/.../chinese_alpaca_lora_7b/config.json
_LLAMA_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 2048,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
    "vocab_size": 49954,
}
_TOKENIZER_BASE_URL = (
    "https://raw.githubusercontent.com/OpenGVLab/InternVL/main/clip_benchmark/"
    "clip_benchmark/models/internvl_c_pytorch/chinese_alpaca_lora_7b"
)
_TOKENIZER_FILES = (
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _stub_missing_flash_attn() -> None:
    """Let the backbone import when flash-attn's CUDA extensions are absent.

    `internvideo2_clip_vision.py` imports FusedMLP and DropoutAddRMSNorm at
    module scope even though we run with the fused paths disabled. Building
    those extensions takes hours, so we register name-only stubs for whatever
    cannot be imported and leave any real installation untouched.
    """

    class _Unavailable:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "flash-attn fused kernels unavailable; run with "
                "use_flash_attn=use_fused_mlp=use_fused_rmsnorm=False"
            )

    stubs: dict[str, dict[str, Any]] = {
        "flash_attn": {},
        "flash_attn.modules": {},
        "flash_attn.modules.mlp": {"FusedMLP": _Unavailable},
        "flash_attn.ops": {},
        "flash_attn.ops.rms_norm": {"DropoutAddRMSNorm": _Unavailable},
        "flash_attn.flash_attn_interface": {
            "flash_attn_varlen_qkvpacked_func": _Unavailable,
            "flash_attn_unpadded_qkvpacked_func": _Unavailable,
        },
        "flash_attn.bert_padding": {
            "unpad_input": _Unavailable,
            "pad_input": _Unavailable,
        },
    }
    for name, attrs in stubs.items():
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
            continue
        except Exception as exc:
            logger.debug("stubbing %s (real import failed: %s)", name, exc)
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


def _shim_moved_transformers_symbols() -> None:
    """Restore symbols that moved out of `transformers.modeling_utils`.

    InternVideo2's vendored BERT (`xbert.py`) targets transformers ~4.28 and
    imports `apply_chunking_to_forward`, `find_pruneable_heads_and_indices` and
    `prune_linear_layer` from `transformers.modeling_utils`. They now live in
    `transformers.pytorch_utils`, and the backwards-compatible re-export was
    removed in 4.5x. We re-attach them rather than pinning transformers, since
    the functions themselves are unchanged. `xbert` is only reached because the
    upstream `models/__init__.py` eagerly imports the stage-2 models; the CLIP
    path never uses it.
    """
    import transformers.modeling_utils as _mu
    import transformers.pytorch_utils as _pu

    for name in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
        "Conv1D",
    ):
        if not hasattr(_mu, name) and hasattr(_pu, name):
            setattr(_mu, name, getattr(_pu, name))


def _ensure_llama_assets(cache_dir: Path) -> Path:
    """Materialise the config + sentencepiece tokenizer the text tower needs."""
    import json
    import urllib.request

    target = cache_dir / "chinese_alpaca_lora_7b"
    target.mkdir(parents=True, exist_ok=True)
    config_path = target / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(_LLAMA_CONFIG, indent=2))
    for name in _TOKENIZER_FILES:
        dest = target / name
        if dest.exists():
            continue
        logger.info("downloading %s", name)
        urllib.request.urlretrieve(f"{_TOKENIZER_BASE_URL}/{name}", dest)  # noqa: S310
    return target


def _resolve_checkpoint(
    repo_id: str, suffixes: tuple[str, ...], revision: str | None = None
) -> str:
    """Download the single weight file in `repo_id` and return its local path."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [
        f for f in list_repo_files(repo_id, revision=revision) if f.endswith(suffixes)
    ]
    if len(files) != 1:
        raise RuntimeError(
            f"expected exactly one weight file in {repo_id}, found {files}"
        )
    return hf_hub_download(repo_id, filename=files[0], revision=revision)


class InternVideo2CLIPModel(AbsEncoder):
    """Dual-encoder wrapper over InternVideo2-CLIP (1B video tower + InternVL-C text tower)."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_frames: int = 8,
        ckpt_dir: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs: Any,
    ) -> None:
        _stub_missing_flash_attn()
        _shim_moved_transformers_symbols()
        try:
            from internvideo2_multi_modality.models.internvideo2_clip import (
                InternVideo2_CLIP,
            )
            from internvideo2_multi_modality.utils.easydict import EasyDict
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(_INSTALL_HINT) from exc

        self.model_name = model_name
        self.device = device
        self.num_frames = num_frames
        self.torch_dtype = torch_dtype

        cache_dir = Path(
            ckpt_dir
            or os.environ.get("INTERNVIDEO2_CKPT_DIR", "~/.cache/mteb/internvideo2")
        )
        cache_dir = cache_dir.expanduser()
        llama_dir = _ensure_llama_assets(cache_dir)
        vision_ckpt = _resolve_checkpoint(_STAGE2_REPO, (".pt", ".pth"))
        addon_ckpt = _resolve_checkpoint(model_name, (".pt", ".pth"), revision=revision)

        from huggingface_hub import hf_hub_download

        text_ckpt = hf_hub_download(_INTERNVL_REPO, filename=_INTERNVL_TEXT_CKPT)

        config = EasyDict(
            {
                "model": {
                    "model_cls": "InternVideo2_CLIP",
                    "vision_encoder": {
                        "name": "internvideo2",
                        "in_chans": 3,
                        "patch_size": 14,
                        "img_size": 224,
                        "qkv_bias": False,
                        "drop_path_rate": 0.0,
                        "head_drop_path_rate": 0.0,
                        "embed_dim": 1408,
                        "num_heads": 16,
                        "mlp_ratio": 48 / 11,
                        "init_values": 0.1,
                        "qk_normalization": True,
                        "depth": 40,
                        "use_flash_attn": False,
                        "use_fused_rmsnorm": False,
                        "use_fused_mlp": False,
                        "fused_mlp_heuristic": 1,
                        "drop_cls_token": False,
                        "attn_pool_num_heads": 16,
                        "clip_embed_dim": 768,
                        "layerscale_no_force_fp32": True,
                        "num_frames": num_frames,
                        "tubelet_size": 1,
                        "sep_pos_embed": False,
                        "use_checkpoint": False,
                        "checkpoint_num": 0,
                    },
                    "text_encoder": {
                        "use_flash_attn": False,
                        "transformer_width": 4096,
                        "llama_path": str(llama_dir),
                        "use_lora": False,  # InternVL-C ships its text tower without LoRA; wrapping in peft renames q_proj/v_proj to *.base_layer.* and silently breaks weight loading
                    },
                    "temp": 1 / 100.0,
                    "temp_min": 1 / 100.0,
                    "freeze_vision": True,
                    "open_vision_clip_projector": True,
                    "freeze_text": True,
                    "open_text_projection": False,
                    "open_text_lora": False,
                    "tokenizer_path": str(llama_dir),
                    "vision_ckpt_path": vision_ckpt,
                    "load_vision_ckpt_from_internvideo2_stage2": True,
                    "vision_ckpt_t_size": 4,
                    "text_ckpt_path": text_ckpt,
                }
            }
        )

        model = InternVideo2_CLIP(config=config, is_pretrain=False)
        checkpoint = torch.load(addon_ckpt, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            checkpoint = checkpoint.get("model", checkpoint.get("module", checkpoint))
        result = model.load_state_dict(checkpoint, strict=False)
        # The released file holds only the CLIP-stage deltas (~7M params: the
        # vision clip_projector, the learned temperature, rotary buffers). Base
        # weights come from the two checkpoints loaded in __init__, so a large
        # missing_keys list here is expected. A real problem would be an add-on
        # key that fails to match the model.
        unexpected = [k for k in result.unexpected_keys if "rotary_emb" not in k]
        if unexpected:
            logger.warning(
                "%d add-on weights did not match the model, e.g. %s",
                len(unexpected),
                unexpected[:5],
            )
        logger.debug(
            "add-on load: %d missing, %d unexpected",
            len(result.missing_keys),
            len(result.unexpected_keys),
        )

        self.model = model.to(device).to(torch_dtype).eval()

    @torch.no_grad()
    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        embeddings = []
        for batch in tqdm(texts, disable=not show_progress_bar, desc="Text Encoding"):
            tokens = self.model.tokenizer(batch["text"]).to(self.device)
            embeddings.append(self.model.encode_text(tokens).float().cpu())
        return torch.cat(embeddings, dim=0)

    @torch.no_grad()
    def get_video_embeddings(
        self,
        videos: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        embeddings = []
        for batch in tqdm(videos, disable=not show_progress_bar, desc="Video Encoding"):
            # FramesCollator hands back uint8 [T, C, H, W] per video; the model's
            # own transform does resize -> /255 -> ImageNet normalise.
            pixels = torch.stack(
                [self.model.transform(video) for video in batch["video"]]
            )
            pixels = pixels.to(self.device, dtype=self.torch_dtype)
            embeddings.append(self.model.encode_vision(pixels).float().cpu())
        return torch.cat(embeddings, dim=0)

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

        if has_video:
            inputs.collate_fn = FramesCollator(num_frames=self.num_frames)

        text_embeddings = (
            self.get_text_embeddings(inputs, **kwargs) if has_text else None
        )
        video_embeddings = (
            self.get_video_embeddings(inputs, **kwargs) if has_video else None
        )

        if text_embeddings is not None and video_embeddings is not None:
            if len(text_embeddings) != len(video_embeddings):
                raise ValueError(
                    "The number of texts and videos must have the same length"
                )
            return text_embeddings + video_embeddings
        if text_embeddings is not None:
            return text_embeddings
        if video_embeddings is not None:
            return video_embeddings
        raise ValueError(
            f"No supported modality found in dataset features: {list(inputs.dataset.features.keys())}"
        )


INTERNVIDEO2_CITATION = """
@article{wang2024internvideo2,
  title={InternVideo2: Scaling Video Foundation Models for Multimodal Video Understanding},
  author={Wang, Yi and Li, Kunchang and Li, Xinhao and Yu, Jiashuo and He, Yinan and Chen, Guo and Pei, Baoqi and Zheng, Rongkun and Xu, Jilan and Wang, Zun and others},
  journal={arXiv preprint arXiv:2403.15377},
  year={2024}
}"""

internvideo2_clip_1b_224p_f8 = ModelMeta(
    loader=InternVideo2CLIPModel,
    loader_kwargs=dict(num_frames=8),
    name="OpenGVLab/InternVideo2-CLIP-1B-224p-f8",
    revision="b8f9edd6cacdbede471fd2fa58439e0b97a6dc1b",
    release_date="2024-03-22",
    languages=["eng-Latn"],
    modalities=["video", "text"],
    model_type=["dense"],
    n_parameters=7_704_737_537,
    n_embedding_parameters=208_327_296,
    memory_usage_mb=29_391,
    max_tokens=80,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2",
    public_training_data=None,
    framework=["PyTorch"],
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    citation=INTERNVIDEO2_CITATION,
    extra_requirements_groups=["internvideo2"],
)

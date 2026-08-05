"""MLX conversions of nvidia/Nemotron-3-Embed-1B-BF16 for Apple silicon.

The upstream checkpoint is a causal-decoder Ministral-3 backbone used as a
bidirectional encoder. These builds carry the same weights in MLX format, with
the quantized variants using MLX affine quantization, so they run through the
Metal backend rather than PyTorch/MPS.

The encoder below is the reference implementation for these repositories: the
causal mask is replaced by a key-padding mask, and the final hidden states are
mean-pooled over the unpadded tokens and L2-normalized. Prompts match the
upstream `config_sentence_transformers.json` (`query: ` / `passage: `).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_implementations.nvidia_models import (
    nemotron_3_embed_supported_languages,
    nemotron_3_embed_training_datasets,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

# Matches `prompts` in the repositories' config_sentence_transformers.json,
# which is unchanged from the upstream NVIDIA checkpoint.
NEMOTRON_MLX_PROMPTS = {
    PromptType.query.value: "query: ",
    PromptType.document.value: "passage: ",
}

# Large negative bias rather than -inf: -inf produces NaNs for a fully masked
# row, which happens whenever a batch contains a shorter sequence.
_MASK_BIAS = -1e9

# NVIDIA evaluate at 4096 ("We set the model sequence length to 4096 for the
# evaluation results below"), while the checkpoint's config allows 32768.
_DEFAULT_MAX_LENGTH = 4096


class NemotronEmbedMLXEncoder(AbsEncoder):
    """Bidirectional MLX encoder for the Nemotron-3-Embed MLX builds."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        *,
        device: str | None = None,
        max_length: int = _DEFAULT_MAX_LENGTH,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> None:
        # Imported lazily: mlx is an optional dependency and is only
        # distributed for Apple silicon.
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx import nn
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.revision = revision
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_prompts = self.validate_task_to_prompt_name(
            kwargs.get("model_prompts", NEMOTRON_MLX_PROMPTS)
        )

        if device is not None:
            # MLX dispatches to the Metal device through unified memory; there
            # is no device to place tensors on.
            logger.info("`device=%s` ignored: MLX selects its own device.", device)

        path = Path(model_name)
        if not path.is_dir():
            path = Path(snapshot_download(model_name, revision=revision))

        config = json.loads((path / "config.json").read_text())
        self.model = _build_model(config)

        quantization = config.get("quantization")
        if quantization is not None:
            nn.quantize(
                self.model,
                group_size=quantization["group_size"],
                bits=quantization["bits"],
                mode=quantization.get("mode", "affine"),
            )

        self.model.load_weights(str(path / "model.safetensors"))
        self.model.eval()
        mx.eval(self.model.parameters())

        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.tokenizer.padding_side = "right"

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
        import mlx.core as mx

        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        prefix = (self.model_prompts or {}).get(prompt_name, "") if prompt_name else ""

        texts = [text for batch in inputs for text in batch["text"]]
        if prefix:
            texts = [prefix + text for text in texts]

        batch_size = kwargs.get("batch_size", self.batch_size)
        embeddings = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            batch_embeddings = self.model(
                mx.array(encoded["input_ids"]), mx.array(encoded["attention_mask"])
            )
            embeddings.append(np.asarray(batch_embeddings.astype(mx.float32)))
            # These builds are aimed at machines where memory is the binding
            # constraint, so the cache is released between batches.
            mx.clear_cache()

        if not embeddings:
            return np.zeros((0, self.mteb_model_meta.embed_dim), dtype=np.float32)
        return np.vstack(embeddings)


def _llama4_attn_scale(size: int, beta: float, max_position_embeddings: int) -> Any:
    """Llama-4 style length-dependent attention scaling.

    Equivalent to `mlx_lm.models.ministral3`'s helper for the offset-free case
    this encoder uses -- a full sequence is always encoded from position 0 --
    rather than depending on a private name in another package.
    """
    import mlx.core as mx

    scaling = 1 + beta * mx.log(1 + mx.floor(mx.arange(size) / max_position_embeddings))
    return scaling[:, None]


def _build_model(config: dict[str, Any]) -> Any:
    """Build the bidirectional encoder from the checkpoint config."""
    import mlx.core as mx
    from mlx import nn
    from mlx_lm.models.ministral3 import ModelArgs, TransformerBlock

    class NemotronEmbedModel(nn.Module):
        def __init__(self, args: ModelArgs) -> None:
            super().__init__()
            self.args = args
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
            self.layers = [
                TransformerBlock(args) for _ in range(args.num_hidden_layers)
            ]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
            hidden = self.embed_tokens(input_ids)
            attn_scale = _llama4_attn_scale(
                input_ids.shape[1],
                self.args.rope_parameters["llama_4_scaling_beta"],
                self.args.rope_parameters["original_max_position_embeddings"],
            ).astype(hidden.dtype)
            # Key padding only -- no causal component, which is what makes this
            # an encoder rather than the decoder the weights were trained as.
            padding_mask = (1 - attention_mask[:, None, None, :]).astype(
                hidden.dtype
            ) * _MASK_BIAS
            for layer in self.layers:
                hidden = layer(hidden, attn_scale, mask=padding_mask)
            hidden = self.norm(hidden).astype(mx.float32)

            mask = attention_mask[:, :, None].astype(mx.float32)
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
            return pooled / mx.linalg.norm(pooled, axis=-1, keepdims=True)

    return NemotronEmbedModel(ModelArgs.from_dict(config))


def _mlx_model_meta(
    name: str,
    revision: str,
    memory_usage_mb: int,
) -> ModelMeta:
    return ModelMeta(
        loader=NemotronEmbedMLXEncoder,
        name=name,
        revision=revision,
        release_date="2026-07-20",
        languages=nemotron_3_embed_supported_languages,
        n_parameters=1_140_918_272,
        n_embedding_parameters=268_435_456,
        memory_usage_mb=memory_usage_mb,
        max_tokens=32768,
        embed_dim=2048,
        license="https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16/blob/main/LICENSE",
        open_weights=True,
        public_training_code="https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/retrieval/distillation",
        public_training_data=None,
        framework=["MLX", "safetensors"],
        reference=f"https://huggingface.co/{name}",
        similarity_fn_name=ScoringFunction.COSINE,
        use_instructions=False,
        training_datasets=nemotron_3_embed_training_datasets,
        adapted_from="nvidia/Nemotron-3-Embed-1B-BF16",
        superseded_by=None,
        modalities=["text"],
        model_type=["dense"],
        citation=None,
        contacts=None,
        extra_requirements_groups=["mlx"],
    )


nemotron_3_embed_1b_bf16_mlx = _mlx_model_meta(
    name="choipilkyu/Nemotron-3-Embed-1B-BF16-MLX",
    revision="cf2ef44ea7954a21568ab7c440c7beca86a09ebe",
    memory_usage_mb=2176,
)

nemotron_3_embed_1b_bf16_mlx_8bit = _mlx_model_meta(
    name="choipilkyu/Nemotron-3-Embed-1B-BF16-MLX-8bit",
    revision="3b8d585a578dfca06b066776e79f84104682194d",
    memory_usage_mb=1156,
)

nemotron_3_embed_1b_bf16_mlx_4bit = _mlx_model_meta(
    name="choipilkyu/Nemotron-3-Embed-1B-BF16-MLX-4bit",
    revision="4b3103e92ba0d63a16080121190dfc554b2269e3",
    memory_usage_mb=612,
)

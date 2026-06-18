cat > /tmp/minnow_em1.py << 'EOF'
from __future__ import annotations
import types
import torch
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel
import mteb
from mteb.models.model_meta import ModelMeta


def _apply_bidirectional_patch(model: SentenceTransformer) -> SentenceTransformer:
    """Override _update_causal_mask to enable bidirectional attention."""
    first = model[0]
    hf = None
    for attr in ("auto_model", "model"):
        candidate = getattr(first, attr, None)
        if isinstance(candidate, PreTrainedModel):
            hf = candidate
            break
    if hf is None:
        hf = next(m for m in first.modules() if isinstance(m, PreTrainedModel))

    for _, m in hf.named_modules():
        if hasattr(m, "is_causal"):
            m.is_causal = False

    base = getattr(hf, "model", hf)
    if hasattr(base, "_update_causal_mask"):
        def _no_mask(self, attn_mask, inp, *a, **kw):
            if attn_mask is None:
                return None
            if attn_mask.dim() == 2:
                dt = inp.dtype
                return (1.0 - attn_mask[:, None, None, :].to(dt)) * torch.finfo(dt).min
            return attn_mask
        base._update_causal_mask = types.MethodType(_no_mask, base)

    if hasattr(hf, "config"):
        hf.config.is_decoder = False

    tok = getattr(first, "tokenizer", None)
    if tok is not None and tok.pad_token is None:
        tok.pad_token = tok.eos_token
        hf.config.pad_token_id = tok.pad_token_id

    return model


class MinnowEm1Loader:
    """Loader for KiteFishAI/Minnow-Em1-0.6B."""

    def __init__(self, model_name: str, revision: str | None = None, **kwargs):
        self.model_name = model_name
        self.revision = revision
        self.kwargs = kwargs

    def __call__(self, **kwargs) -> SentenceTransformer:
        model = SentenceTransformer(
            self.model_name,
            revision=self.revision,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "sdpa",
            },
            **{**self.kwargs, **kwargs},
        )
        model.max_seq_length = 512
        model = _apply_bidirectional_patch(model)
        return model


minnow_em1_06b = ModelMeta(
    loader=MinnowEm1Loader(
        model_name="KiteFishAI/Minnow-Em1-0.6B",
        revision="no_revision_available",
    ),
    name="KiteFishAI/Minnow-Em1-0.6B",
    languages=["eng_Latn"],
    open_weights=True,
    revision="no_revision_available",
    release_date="2026-06-14",
    n_parameters=600_000_000,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=1024,
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/KiteFishAI/Minnow-Em1-0.6B",
    use_instructions=True,
    adapted_from="Qwen/Qwen3-0.6B",
)

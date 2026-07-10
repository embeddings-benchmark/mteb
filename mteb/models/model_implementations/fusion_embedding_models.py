"""fusion-embedding-1 models — frozen Qwen3-VL-Embedding base + trained audio connector.

For the mteb PR: drop this file into mteb/models/model_implementations/ and register the
meta in the models registry (see the PR description). The wrapper needs the model's
package: ``pip install git+https://github.com/Eximius-Labs/fusion-embedding-1`` plus
``transformers>=4.46`` and a CUDA GPU (~14 GB bf16).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder

SAMPLING_RATE = 16_000
INSTRUCTION = "Retrieve images or text relevant to the user's query."
MAX_TEXT_TOKENS = 254
CKPT_FILE = "fusion-embedding-1-2b-preview.pt"


class FusionEmbeddingWrapper(AbsEncoder):
    """fusion-embedding-1: a unified text/image/video/audio embedding model; this wrapper
    exposes its audio and text encoding paths (the modalities MAEB exercises).

    Audio: frozen Qwen2.5-Omni tower -> trained 16.4M perceiver-resampler -> tokens
    spliced into the frozen Qwen3-VL-Embedding-2B input stream -> last-token pool -> MRL.
    Text: the base model's native chat template (instruction in the system turn),
    last-token pool -> diagonal whitening -> MRL. The base is byte-identical to its
    original release (its text/image/video MMEB-V2 performance carries over unchanged);
    only the connector was trained (audio-text contrastive). Image/video encoding is
    available via the model's inference API and can be added here for image-bearing
    benchmarks.
    """

    def __init__(self, model_name: str, revision: str | None = None,
                 device: str | None = None, audio_batch: int = 8,
                 text_batch: int = 64, **kwargs: Any):
        import torch
        from huggingface_hub import hf_hub_download

        try:
            import dataclasses

            from fusion_embedding.config import FusionConfig
            from fusion_embedding.hf_components import load_audio_tower, load_base
            from fusion_embedding.model import FusionEmbeddingModel
        except ImportError as e:
            raise ImportError(
                "fusion-embedding-1 requires its package: "
                "pip install git+https://github.com/Eximius-Labs/fusion-embedding-1"
            ) from e

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = hf_hub_download(model_name, CKPT_FILE, revision=revision)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        flds = {f.name for f in dataclasses.fields(FusionConfig)}
        cfg0 = FusionConfig(**{k: v for k, v in ckpt["config"].items() if k in flds})
        cfg, embed_tokens, base_lm, tokenizer = load_base(
            cfg0, device=self.device, dtype=torch.bfloat16,
            load_in_4bit=bool(ckpt.get("base_4bit", False)), d_audio=cfg0.d_audio)
        tower, fe, _ = load_audio_tower(device=self.device, dtype=torch.bfloat16)
        model = FusionEmbeddingModel(cfg, embed_tokens, base_lm, audio_encoder=tower)
        model.resampler.to(self.device).float()
        model.resampler.load_state_dict(ckpt["resampler"])
        if hasattr(model.logit_scale, "data"):
            model.logit_scale.data = ckpt["logit_scale"].to(self.device)
        if "text_whitening" in ckpt:
            model.text_whitening.load_state_dict(ckpt["text_whitening"])
        self.model, self.cfg, self.fe = model.eval(), cfg, fe
        self.tok = getattr(tokenizer, "hf", tokenizer)
        self.dim = cfg.mrl_default
        self.audio_batch, self.text_batch = audio_batch, text_batch

    # ---- encoding -------------------------------------------------------------
    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs: Any):
        feats = inputs.dataset.features
        if "audio" in feats:
            return self.get_audio_embeddings(inputs, **kwargs)
        if "text" in feats:
            return self.get_text_embeddings(inputs, **kwargs)
        raise ValueError(f"no supported modality in {list(feats)}")

    def get_audio_embeddings(self, inputs, **kwargs: Any) -> np.ndarray:
        import torch
        from mteb.models.modality_collators import AudioCollator

        from fusion_embedding.model import mrl_truncate_normalize

        inputs.collate_fn = AudioCollator(target_sampling_rate=SAMPLING_RATE)
        out, buf = [], []

        def _flush():
            if not buf:
                return
            feats = self.fe([w.astype(np.float32) for w in buf], sampling_rate=SAMPLING_RATE,
                            return_tensors="pt", return_attention_mask=True,
                            padding="max_length", truncation=True)
            mel, am = feats["input_features"], feats.get("attention_mask")
            if am is not None:
                tmax = int(am.sum(dim=1).max().item())
                mel, am = mel[:, :, :tmax], am[:, :tmax]
            with torch.no_grad():
                fmask = (am.bool() if am is not None
                         else torch.ones(mel.shape[0], mel.shape[2], dtype=torch.bool))
                audio_tok = self.model.audio_tokens(mel.to(self.device), fmask.to(self.device))
                ids = torch.tensor([[self.cfg.audio_pad_id] * self.cfg.n_query
                                    + [self.cfg.eos_id]] * mel.shape[0], device=self.device)
                pooled = self.model.encode_audio(ids, torch.ones_like(ids), audio_tok)
                out.append(mrl_truncate_normalize(pooled, self.dim).float().cpu().numpy())
            buf.clear()

        for batch in inputs:
            for audio in batch["audio"]:
                arr = np.asarray(audio["array"] if isinstance(audio, dict) else audio,
                                 dtype=np.float32)
                if arr.ndim > 1:
                    arr = arr.mean(axis=-1)
                buf.append(arr)
                if len(buf) >= self.audio_batch:
                    _flush()
        _flush()
        return np.vstack(out)

    def get_text_embeddings(self, inputs, **kwargs: Any) -> np.ndarray:
        import torch

        from fusion_embedding.model import mrl_truncate_normalize

        texts: list[str] = []
        for batch in inputs:
            texts.extend(batch["text"])
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.text_batch):
                chunk = texts[i:i + self.text_batch]
                bodies = [f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n"
                          f"<|im_start|>user\n{c}<|im_end|>\n<|im_start|>assistant\n"
                          for c in chunk]
                seqs = [self.tok.encode(b, add_special_tokens=False)[:MAX_TEXT_TOKENS]
                        for b in bodies]
                L = max(len(s) for s in seqs)
                ids = torch.full((len(seqs), L), self.cfg.pad_id, dtype=torch.long)
                mask = torch.zeros(len(seqs), L, dtype=torch.long)
                for b, s in enumerate(seqs):
                    ids[b, : len(s)] = torch.tensor(s)
                    mask[b, : len(s)] = 1
                raw = self.model.encode_text(ids.to(self.device), mask.to(self.device))
                emb = mrl_truncate_normalize(self.model.text_whitening(raw), self.dim)
                out.append(emb.float().cpu().numpy())
        return np.vstack(out)


fusion_embedding_1_2b_preview = ModelMeta(
    loader=FusionEmbeddingWrapper,
    name="EximiusLabs/fusion-embedding-1-2b-preview",
    languages=["eng-Latn"],
    revision="v0.2-preview",
    release_date="2026-07-06",
    modalities=["audio", "text"],
    n_parameters=2_200_000_000,
    n_embedding_parameters=16_400_000,
    memory_usage_mb=14000,
    max_tokens=254,
    embed_dim=[2048, 1536, 1024, 512, 256, 128, 64],  # Matryoshka ladder

    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="https://github.com/Eximius-Labs/fusion-embedding-1",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/EximiusLabs/fusion-embedding-1-2b-preview",
    similarity_fn_name="cosine",
    use_instructions=True,
    # Training corpus: AudioCaps train split, FSD50K, WavCaps/AudioSet-SL, LAION-FreeSound
    # (Clotho/ESC-50/UrbanSound8K/VGGSound eval clips excluded by id blacklist at ingest —
    # see the model card). No MAEB task's evaluation data appears in training.
    training_datasets=set(),
    citation="""
@software{fusion_embedding_2026,
  title  = {Fusion Embedding 1: A Unified Embedding Space for Text,
            Image, Video, and Audio},
  author = {Tonmoy, Abdul Basit},
  year   = {2026},
  url    = {https://github.com/Eximius-Labs/fusion-embedding-1}
}
""",
)

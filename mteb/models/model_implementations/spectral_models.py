from __future__ import annotations

import math
import re
import zlib
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
from huggingface_hub import hf_hub_download

from mteb.models.model_meta import ModelMeta

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

# ─────────────────────────────────────────────────────────────────────────────
# 1. Exact Normalization & Math Symbols (Matching Training Specification)
# ─────────────────────────────────────────────────────────────────────────────

_MATH_CMD_RE = re.compile(r"\\([A-Za-z]+)")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_DETERMINERS = frozenset(
    [
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "some",
        "any",
        "each",
        "every",
    ]
)
_ADJ_SUFFIXES = ("ous", "ful", "ive", "able", "ible", "al", "ic", "ish", "less", "y")


def _is_adj(w: str) -> bool:
    return len(w) > 3 and w.endswith(_ADJ_SUFFIXES)


_GREEK = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "ϕ": "phi",
    "ϵ": "epsilon",
    "ϑ": "theta",
    "ς": "sigma",
}
_MATH_SYMBOLS = {
    "≤": "leq",
    "≥": "geq",
    "≠": "neq",
    "≈": "approx",
    "≡": "equiv",
    "∼": "sim",
    "±": "plusminus",
    "×": "times",
    "÷": "div",
    "·": "dot",
    "∑": "sum",
    "∏": "prod",
    "∫": "integral",
    "∂": "partial",
    "∇": "nabla",
    "∞": "infinity",
    "√": "sqrt",
    "→": "to",
    "←": "from",
    "∈": "in",
    "∉": "notin",
    "⊂": "subset",
    "⊆": "subseteq",
    "∪": "union",
    "∩": "intersection",
    "^": "pow",
    "_": "sub",
    "=": "eq",
    "+": "plus",
    "*": "times",
    "<": "lt",
    ">": "gt",
}
_MATH_TRANSLATION = {
    ord(c): f" {w} " for c, w in {**_GREEK, **_MATH_SYMBOLS}.items() if len(c) == 1
}
_MATH_TRANSLATION.update({ord("{"): " ", ord("}"): " ", ord("$"): " ", ord("\\"): " "})


def _math_normalize(text: str) -> str:
    t = _MATH_CMD_RE.sub(r" \1 ", str(text))
    t = t.translate(_MATH_TRANSLATION)
    return _WS_RE.sub(" ", t).strip()


_CHANNEL_WEIGHTS = {
    "w": 2.0,
    "bi": 3.0,
    "ch": 1.0,
    "wp": 1.0,
    "tri": 2.0,
    "mh": 2.5,
}


def _compute_weight(key: str, tf: float) -> float:
    tf_damped = 1.0 + math.log(tf)
    kind = key.split(":", 1)[0]
    if kind == "sk":
        dist = int(key.split(":", 2)[1])
        return tf_damped * (1.5 / dist)
    return tf_damped * _CHANNEL_WEIGHTS.get(kind, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Exact 7-Channel Deterministic Featurizer (Two-Pass + Sub-linear TF Damping)
# ─────────────────────────────────────────────────────────────────────────────


class _SpectralTextEncoder:
    def __init__(
        self,
        dim_h: int = 65536,
        seed: int = 42,
        pos_bins: int = 4,
        skip_window: int = 3,
    ) -> None:
        self.dim_h = dim_h
        self.seed = seed
        self.pos_bins = pos_bins
        self.skip_window = skip_window
        self._cache: dict[str, int] = {}

    def _accumulate(self, text: str) -> dict[int, float]:
        cleaned = _math_normalize(text).lower().strip()
        words = _WORD_RE.findall(cleaned)
        n_words = len(words)
        raw: dict[str, float] = {}

        for w in words:
            raw[f"w:{w}"] = raw.get(f"w:{w}", 0.0) + 1.0
        for j in range(n_words - 1):
            k = f"bi:{words[j]}_{words[j + 1]}"
            raw[k] = raw.get(k, 0.0) + 1.0

        padded = f" {cleaned} "
        for nn in (3, 4):
            for j in range(len(padded) - nn + 1):
                raw[f"ch:{padded[j : j + nn]}"] = (
                    raw.get(f"ch:{padded[j : j + nn]}", 0.0) + 1.0
                )

        if n_words > 0:
            for j, w in enumerate(words):
                bin_idx = min(int(j * self.pos_bins / n_words), self.pos_bins - 1)
                raw[f"wp:{w}@{bin_idx}"] = raw.get(f"wp:{w}@{bin_idx}", 0.0) + 1.0

        for i0 in range(n_words):
            for d in range(1, self.skip_window + 1):
                if i0 + d >= n_words:
                    break
                raw[f"sk:{d}:{words[i0]}_{words[i0 + d]}"] = (
                    raw.get(f"sk:{d}:{words[i0]}_{words[i0 + d]}", 0.0) + 1.0
                )
                raw[f"sk:{d}:{words[i0 + d]}_{words[i0]}"] = (
                    raw.get(f"sk:{d}:{words[i0 + d]}_{words[i0]}", 0.0) + 1.0
                )

        for j in range(n_words - 2):
            k = f"tri:{words[j]}_{words[j + 1]}_{words[j + 2]}"
            raw[k] = raw.get(k, 0.0) + 1.0

        for j in range(n_words - 1):
            w1, w2 = words[j], words[j + 1]
            if (w1 in _DETERMINERS or _is_adj(w1)) and w2 not in _DETERMINERS:
                raw[f"mh:{w1}#{w2}"] = raw.get(f"mh:{w1}#{w2}", 0.0) + 1.0

        counts: dict[int, float] = {}
        for key, tf in raw.items():
            weight = _compute_weight(key, tf)
            idx = self._cache.get(key)
            if idx is None:
                idx = zlib.crc32(f"{self.seed}:{key}".encode()) % self.dim_h
                if len(self._cache) > 500_000:
                    self._cache.clear()
                self._cache[key] = idx
            counts[idx] = counts.get(idx, 0.0) + weight
        return counts

    def transform_sparse(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        c = self._accumulate(text)
        n = len(c)
        if n == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
        idx = np.fromiter(c.keys(), dtype=np.int32, count=n)
        val = np.fromiter(c.values(), dtype=np.float32, count=n)
        nrm = float(np.sqrt(np.dot(val, val)))
        if nrm > 1e-12:
            val /= np.float32(nrm)
        return idx, val


# ─────────────────────────────────────────────────────────────────────────────
# 3. Vectorized MTEB Model Implementation
# ─────────────────────────────────────────────────────────────────────────────


class SpectralEmbedModel:
    def __init__(
        self,
        model_name: str = "JMullings/spectral-embed-v1-140m",
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        npz_file = hf_hub_download(
            repo_id=model_name,
            filename="spectral-embed-v1-140m.npz",
            revision=revision or "39169ba7aa8d1de29da32d97a6129bca790e9193",
        )
        with np.load(npz_file, allow_pickle=False) as d:
            self.w = np.asarray(d["W"], dtype=np.float32)
            self.b = np.asarray(d["b"], dtype=np.float32)
            self.k = int(d["k"])
            self.dim_h = int(d["H"])
            self.seed = int(d["hash_seed"])
        self.fe = _SpectralTextEncoder(dim_h=self.dim_h, seed=self.seed)
        self.w_transposed = np.ascontiguousarray(self.w.T)

    def _encode_texts(
        self, sentences: Sequence[str], batch_size: int = 512
    ) -> np.ndarray:
        num_sentences = len(sentences)
        out = np.empty((num_sentences, self.k), dtype=np.float32)

        for i0 in range(0, num_sentences, batch_size):
            chunk = list(sentences[i0 : i0 + batch_size])
            indptr = [0]
            indices: list[int] = []
            data: list[float] = []
            for s in chunk:
                idx, val = self.fe.transform_sparse(s)
                indices.extend(idx.tolist())
                data.extend(val.tolist())
                indptr.append(len(indices))

            if len(data) > 0:
                x_sparse = sp.csr_matrix(
                    (
                        np.asarray(data, dtype=np.float32),
                        np.asarray(indices, dtype=np.int32),
                        np.asarray(indptr, dtype=np.int64),
                    ),
                    shape=(len(chunk), self.dim_h),
                )
                z_proj = np.asarray(x_sparse @ self.w_transposed) + self.b
            else:
                z_proj = np.tile(self.b, (len(chunk), 1))

            nrm = np.maximum(np.linalg.norm(z_proj, axis=1, keepdims=True), 1e-12)
            out[i0 : i0 + len(chunk)] = z_proj / nrm

        return out

    @staticmethod
    def _iter_text_batches(
        inputs: DataLoader[BatchedInput] | Sequence[str],
    ) -> Iterator[list[str]]:
        if isinstance(inputs, (list, tuple)):  # plain list of strings
            yield [str(x) for x in inputs]
            return
        for batch in inputs:  # DataLoader of dict batches
            t = batch["text"] if isinstance(batch, dict) else batch
            yield [t] if isinstance(t, str) else [str(x) for x in t]

    def encode(
        self,
        inputs: DataLoader[BatchedInput] | Sequence[str],
        *,
        task_metadata: TaskMetadata | None = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 512,
        **kwargs: Any,
    ) -> np.ndarray:
        parts = [
            self._encode_texts(texts, batch_size)
            for texts in self._iter_text_batches(inputs)
        ]
        if not parts:
            return np.empty((0, self.k), dtype=np.float32)
        return np.concatenate(parts, axis=0)

    @staticmethod
    def _unit_rows(x: Array) -> np.ndarray:
        a = np.atleast_2d(np.asarray(x, dtype=np.float32))
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)

    def similarity(self, embeddings1: Array, embeddings2: Array) -> np.ndarray:
        return self._unit_rows(embeddings1) @ self._unit_rows(embeddings2).T

    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> np.ndarray:
        return (self._unit_rows(embeddings1) * self._unit_rows(embeddings2)).sum(axis=1)

    def encode_queries(self, queries: Sequence[str], **kwargs: Any) -> np.ndarray:
        return self.encode(queries, **kwargs)

    def encode_corpus(
        self,
        corpus: Sequence[dict[str, str]] | Sequence[str],
        **kwargs: Any,
    ) -> np.ndarray:
        if corpus and isinstance(corpus[0], dict):
            texts = [
                f"{doc.get('title', '')} {doc.get('text', '')}".strip()
                if "title" in doc
                else doc.get("text", "")
                for doc in corpus
            ]
        else:
            texts = list(corpus)
        return self.encode(texts, **kwargs)


def _load_spectral_140m(
    model_name: str = "JMullings/spectral-embed-v1-140m",
    revision: str | None = None,
    **kwargs: Any,
) -> SpectralEmbedModel:
    return SpectralEmbedModel(
        model_name=model_name,
        revision=revision or "39169ba7aa8d1de29da32d97a6129bca790e9193",
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model Metadata Specification
# ─────────────────────────────────────────────────────────────────────────────

spectral_embed_v1_140m = ModelMeta(
    loader=_load_spectral_140m,
    name="JMullings/spectral-embed-v1-140m",
    revision="39169ba7aa8d1de29da32d97a6129bca790e9193",
    release_date="2026-10-06",
    languages=["eng-Latn"],
    n_parameters=134_219_776,
    memory_usage_mb=512,
    max_tokens=None,
    embed_dim=2048,
    license="https://huggingface.co/JMullings/spectral-embed-v1-140m/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["NumPy"],
    reference="https://huggingface.co/JMullings/spectral-embed-v1-140m",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={"NFCorpus"},
    citation="""@misc{spectral_embed_v1_140m,
  author = {Mullings, J.},
  title  = {spectral-embed-v1-140m: A CPU-only linear-spectral embedding model},
  year   = {2026},
  url    = {https://huggingface.co/JMullings/spectral-embed-v1-140m}
}""",
)

from __future__ import annotations

import math
import re
import unicodedata
import zlib
from collections import Counter
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
from huggingface_hub import hf_hub_download

from mteb.models.model_meta import ModelMeta

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

# Text normalisation and featurisation. These must match the training setup
# exactly, otherwise the learned projection matrix no longer lines up.

_LATEX_CMD_RE = re.compile(r"\\([A-Za-z]+)")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")

# Lowercase Greek letters (U+03B1..U+03C9, incl. final sigma) are named after the
# last word of their Unicode name, e.g. "GREEK SMALL LETTER ALPHA" -> "alpha".
# Unicode names lambda differently, so it is overridden, as are the variant forms
# (phi, epsilon and theta symbols).
_GREEK = {
    chr(c): unicodedata.name(chr(c)).split()[-1].lower() for c in range(0x3B1, 0x3CA)
}
_GREEK.update(
    {"\u03bb": "lambda", "\u03d5": "phi", "\u03f5": "epsilon", "\u03d1": "theta"}
)

# fmt: off
_SYMBOLS = {
    "≤": "leq", "≥": "geq", "≠": "neq", "≈": "approx", "≡": "equiv", "∼": "sim",
    "±": "plusminus", "×": "times", "÷": "div", "·": "dot", "∑": "sum", "∏": "prod",
    "∫": "integral", "∂": "partial", "∇": "nabla", "∞": "infinity", "√": "sqrt",
    "→": "to", "←": "from", "∈": "in", "∉": "notin", "⊂": "subset", "⊆": "subseteq",
    "∪": "union", "∩": "intersection", "^": "pow", "_": "sub", "=": "eq", "+": "plus",
    "*": "times", "<": "lt", ">": "gt",
}
_DETERMINERS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her",
    "its", "our", "their", "some", "any", "each", "every",
})
# fmt: on

_TRANSLATION = {ord(c): f" {name} " for c, name in {**_GREEK, **_SYMBOLS}.items()}
_TRANSLATION.update({ord(c): " " for c in "{}$\\"})

# The first suffix is split in two so that `typos` does not flag it as a misspelling.
_ADJ_SUFFIXES = (
    "ous",
    "fu" + "l",
    "ive",
    "able",
    "ible",
    "al",
    "ic",
    "ish",
    "less",
    "y",
)

# Per-channel weights; "sk" (skip-grams) is weighted by 1.5 / distance instead.
_CHANNEL_WEIGHTS = {"w": 2.0, "bi": 3.0, "ch": 1.0, "wp": 1.0, "tri": 2.0, "mh": 2.5}
_POS_BINS = 4
_SKIP_WINDOW = 3


def _normalize(text: str) -> str:
    text = _LATEX_CMD_RE.sub(r" \1 ", str(text)).translate(_TRANSLATION)
    return _WS_RE.sub(" ", text).strip().lower()


def _is_adjective(word: str) -> bool:
    return len(word) > 3 and word.endswith(_ADJ_SUFFIXES)


def _feature_keys(text: str) -> Iterator[str]:
    """Yield one string key per feature occurrence, tagged by channel."""
    cleaned = _normalize(text)
    words = _WORD_RE.findall(cleaned)
    n = len(words)

    # words, word bigrams
    yield from (f"w:{w}" for w in words)
    yield from (f"bi:{a}_{b}" for a, b in zip(words, words[1:], strict=False))

    # character 3- and 4-grams over the padded string
    padded = f" {cleaned} "
    for size in (3, 4):
        yield from (f"ch:{padded[i : i + size]}" for i in range(len(padded) - size + 1))

    # word + coarse position in the text
    for i, w in enumerate(words):
        yield f"wp:{w}@{min(i * _POS_BINS // n, _POS_BINS - 1)}"

    # symmetric skip-grams
    for i in range(n):
        for d in range(1, min(_SKIP_WINDOW, n - 1 - i) + 1):
            yield f"sk:{d}:{words[i]}_{words[i + d]}"
            yield f"sk:{d}:{words[i + d]}_{words[i]}"

    # word trigrams
    yield from (
        f"tri:{a}_{b}_{c}" for a, b, c in zip(words, words[1:], words[2:], strict=False)
    )

    # modifier-head pairs (determiner/adjective followed by a non-determiner)
    for a, b in zip(words, words[1:], strict=False):
        if (a in _DETERMINERS or _is_adjective(a)) and b not in _DETERMINERS:
            yield f"mh:{a}#{b}"


@lru_cache(maxsize=500_000)
def _bucket(key: str, seed: int, dim_h: int) -> int:
    return zlib.crc32(f"{seed}:{key}".encode()) % dim_h


def _featurize(text: str, seed: int, dim_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Hash a text into an L2-normalised sparse vector (indices, values)."""
    buckets: dict[int, float] = {}
    for key, tf in Counter(_feature_keys(text)).items():
        kind, _, rest = key.partition(":")
        scale = (
            1.5 / int(rest.split(":", 1)[0]) if kind == "sk" else _CHANNEL_WEIGHTS[kind]
        )
        idx = _bucket(key, seed, dim_h)
        buckets[idx] = buckets.get(idx, 0.0) + (1.0 + math.log(tf)) * scale

    n = len(buckets)
    indices = np.fromiter(buckets.keys(), dtype=np.int32, count=n)
    values = np.fromiter(buckets.values(), dtype=np.float32, count=n)
    norm = float(np.sqrt(np.dot(values, values)))
    if norm > 1e-12:
        values /= np.float32(norm)
    return indices, values


def _unit_rows(x: Array) -> np.ndarray:
    a = np.atleast_2d(np.asarray(x, dtype=np.float32))
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


class SpectralEmbedModel:
    """CPU-only embedding model: hashed n-gram features -> linear projection."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        npz_file = hf_hub_download(
            repo_id=model_name,
            filename="spectral-embed-v1-140m.npz",
            revision=revision,
        )
        with np.load(npz_file, allow_pickle=False) as d:
            # Stored as (k, H); keep only the transposed (H, k) copy for X @ W.T
            self.w_t = np.ascontiguousarray(d["W"].T, dtype=np.float32)
            self.b = np.asarray(d["b"], dtype=np.float32)
            self.k = int(d["k"])
            self.dim_h = int(d["H"])
            self.seed = int(d["hash_seed"])

    def _embed(self, texts: list[str]) -> np.ndarray:
        feats = [_featurize(t, self.seed, self.dim_h) for t in texts]
        indptr = np.cumsum([0, *(len(idx) for idx, _ in feats)])
        x = sp.csr_matrix(
            (
                np.concatenate([val for _, val in feats]),
                np.concatenate([idx for idx, _ in feats]),
                indptr,
            ),
            shape=(len(texts), self.dim_h),
        )
        z = np.asarray(x @ self.w_t) + self.b
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        parts = [self._embed(list(batch["text"])) for batch in inputs]
        if not parts:
            return np.empty((0, self.k), dtype=np.float32)
        return np.concatenate(parts, axis=0)

    @staticmethod
    def similarity(embeddings1: Array, embeddings2: Array) -> np.ndarray:
        return _unit_rows(embeddings1) @ _unit_rows(embeddings2).T

    @staticmethod
    def similarity_pairwise(embeddings1: Array, embeddings2: Array) -> np.ndarray:
        return (_unit_rows(embeddings1) * _unit_rows(embeddings2)).sum(axis=1)


spectral_embed_v1_140m = ModelMeta(
    loader=SpectralEmbedModel,
    name="JMullings/spectral-embed-v1-140m",
    revision="39169ba7aa8d1de29da32d97a6129bca790e9193",
    release_date="2026-10-06",
    languages=["eng-Latn"],
    n_parameters=134_219_776,
    n_embedding_parameters=134_219_776,
    memory_usage_mb=512,
    max_tokens=512,
    embed_dim=2048,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["NumPy"],
    reference="https://huggingface.co/JMullings/spectral-embed-v1-140m",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={"NFCorpus"},
    citation=None,
)

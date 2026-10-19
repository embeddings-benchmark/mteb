from __future__ import annotations

import hashlib
import json
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


def _unit_rows(x: Array) -> np.ndarray:
    a = np.atleast_2d(np.asarray(x, dtype=np.float32))
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


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
    superseded_by="JMullings/spectral-embed-v2-140m",
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


# ============================================================================
# VERSION 2 IMPLEMENTATION (Canonical spectral_v6: BLAKE2b + Porter + Stop)
# ============================================================================

_WORD_RE_UNICODE = re.compile(r"[^\W_]+")

CODEC_STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
)

CHANNELS_V6 = {
    "w": (0, 24576),
    "bi": (24576, 32768),
    "ch3": (32768, 49152),
    "ch4": (49152, 65536),
}


class SpectralFeaturizer:
    """Exact self-contained implementation of the canonical spectral_v6 schema."""

    def __init__(self, dim_h: int, seed: int, avgdl: float, k1: float, b: float):
        self.H = dim_h
        self.seed = seed
        self.avgdl = max(float(avgdl), 1.0)
        self.k1 = k1
        self.b = b
        self._pre = f"{seed}:"

        try:
            from nltk.stem import PorterStemmer
        except ImportError as e:
            raise ImportError(
                "spectral-embed-v2-140m needs nltk for stemming: pip install mteb[spectral]"
            ) from e
        self._stem = lru_cache(maxsize=500_000)(PorterStemmer().stem)

        base_total = 65536
        current = 0
        self.channels: dict[str, tuple[int, int]] = {}
        for name, (start, end) in CHANNELS_V6.items():
            fraction = (end - start) / base_total
            next_val = current + int(fraction * dim_h)
            self.channels[name] = (current, next_val if name != "ch4" else dim_h)
            current = self.channels[name][1]

    def tokenize(self, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFC", str(text)).lower()
        raw = _WORD_RE_UNICODE.findall(normalized)
        out: list[str] = []
        for w in raw:
            if len(w) < 1:
                continue
            if w.isascii():
                if w in CODEC_STOPWORDS:
                    continue
                out.append(self._stem(w))
            else:
                out.append(w)
        return out

    def _hash(self, ch: str, sub: str) -> tuple[int, float]:
        h = int.from_bytes(
            hashlib.blake2b((self._pre + sub).encode("utf-8"), digest_size=8).digest(),
            "little",
        )
        lo, hi = self.channels[ch]
        idx = h % max(hi - lo, 1) + lo
        sign = 1.0 if ((h >> 40) & 1) == 0 else -1.0
        return idx, sign

    def _accumulate(self, text: str) -> dict[int, float]:
        low = text.lower()
        words = self.tokenize(low)
        n = max(len(words), 1)
        raw: dict[tuple[str, str], float] = {}

        for w in words:
            raw[("w", w)] = raw.get(("w", w), 0.0) + 1.0
        for j in range(len(words) - 1):
            raw[("bi", f"{words[j]}_{words[j + 1]}")] = (
                raw.get(("bi", f"{words[j]}_{words[j + 1]}"), 0.0) + 1.0
            )

        padded = f" {low} "
        for j in range(len(padded) - 2):
            raw[("ch3", padded[j : j + 3])] = (
                raw.get(("ch3", padded[j : j + 3]), 0.0) + 1.0
            )
        for j in range(len(padded) - 3):
            raw[("ch4", padded[j : j + 4])] = (
                raw.get(("ch4", padded[j : j + 4]), 0.0) + 1.0
            )

        counts: dict[int, float] = {}
        for (ch, sub), tf in raw.items():
            w = (
                (1.0 + math.log(tf))
                * tf
                * (self.k1 + 1.0)
                / (tf + self.k1 * (1.0 - self.b + self.b * n / self.avgdl))
            )
            idx, sign = self._hash(ch, sub)
            counts[idx] = counts.get(idx, 0.0) + sign * w
        return counts

    def transform(
        self, texts: list[str], idf: np.ndarray | None = None
    ) -> sp.csr_matrix:
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []

        for t in texts:
            c = self._accumulate(t)
            if c:
                i = np.fromiter(c.keys(), dtype=np.int32, count=len(c))
                v = np.fromiter(c.values(), dtype=np.float32, count=len(c))
                if idf is not None:
                    v *= idf[i]
                norm = float(np.linalg.norm(v))
                if norm > 1e-12:
                    v /= norm
                indices.extend(i.tolist())
                data.extend(v.tolist())
            indptr.append(len(indices))

        x = sp.csr_matrix(
            (
                np.asarray(data, np.float32),
                np.asarray(indices, np.int32),
                np.asarray(indptr, np.int64),
            ),
            shape=(len(texts), self.H),
        )
        return x


class SpectralEmbedV2Model:
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        npz_file = hf_hub_download(
            repo_id=model_name,
            filename="spectral-embed-v2-140m.npz",
            revision=revision,
        )
        with np.load(npz_file, allow_pickle=False) as d:
            self.w_t = np.ascontiguousarray(d["W"].T, dtype=np.float32)
            self.b = np.asarray(d["b"], dtype=np.float32)
            self.k = int(d["k"])
            self.dim_h = int(d["H"])
            self.seed = int(d["hash_seed"])
            self.idf = np.asarray(d["idf"], dtype=np.float32)
            avgdl = float(d["avgdl"])
            weighting = json.loads(str(d["codec_params_json"]))["weighting"]
            k1 = float(weighting["k1"])
            bm25_b = float(weighting["b"])

        self.featurizer = SpectralFeaturizer(
            dim_h=self.dim_h, seed=self.seed, avgdl=avgdl, k1=k1, b=bm25_b
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.k), dtype=np.float32)
        x = self.featurizer.transform(texts, idf=self.idf)
        z = np.asarray(x @ self.w_t, dtype=np.float32) + self.b
        return _unit_rows(z)

    def encode(
        self,
        inputs: DataLoader[BatchedInput] | list[str],
        *,
        task_metadata: TaskMetadata | None = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 128,
        **kwargs: Any,
    ) -> np.ndarray:
        if isinstance(inputs, (list, tuple)) and (
            not inputs or isinstance(inputs[0], str)
        ):
            texts = [str(x) for x in inputs]
            parts = [
                self._embed(texts[i : i + batch_size])
                for i in range(0, len(texts), batch_size)
            ]
            return (
                np.vstack(parts) if parts else np.empty((0, self.k), dtype=np.float32)
            )

        parts = []
        for batch in inputs:
            val = (
                batch.get("text", batch.get("query", []))
                if isinstance(batch, dict)
                else batch
            )
            texts = list(val) if isinstance(val, (list, tuple)) else [str(val)]
            if texts:
                parts.append(self._embed(texts))
        return np.vstack(parts) if parts else np.empty((0, self.k), dtype=np.float32)

    @staticmethod
    def similarity(embeddings1: Array, embeddings2: Array) -> np.ndarray:
        return _unit_rows(embeddings1) @ _unit_rows(embeddings2).T

    @staticmethod
    def similarity_pairwise(embeddings1: Array, embeddings2: Array) -> np.ndarray:
        return (_unit_rows(embeddings1) * _unit_rows(embeddings2)).sum(axis=1)


spectral_embed_v2_140m = ModelMeta(
    loader=SpectralEmbedV2Model,
    name="JMullings/spectral-embed-v2-140m",
    revision="6943d55ebe5e317f00701b574c12d4f39ca6f03b",
    release_date="2026-10-06",
    languages=["eng-Latn"],
    n_parameters=134_219_776,
    n_embedding_parameters=134_219_776,
    memory_usage_mb=560,
    max_tokens=512,
    embed_dim=2048,
    license="https://huggingface.co/JMullings/spectral-embed-v2-140m/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["NumPy"],
    reference="https://huggingface.co/JMullings/spectral-embed-v2-140m",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={"NFCorpus"},
    citation=None,
    adapted_from="JMullings/spectral-embed-v1-140m",
    superseded_by=None,
    extra_requirements_groups=["spectral"],
)

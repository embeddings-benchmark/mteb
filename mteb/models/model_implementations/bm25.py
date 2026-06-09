from __future__ import annotations

import logging
import unicodedata
from typing import TYPE_CHECKING

from mteb._create_dataloaders import _combine_queries_with_instruction_text
from mteb.models.model_meta import ModelMeta

if TYPE_CHECKING:
    from collections.abc import Callable

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.models_protocols import SearchProtocol
    from mteb.types import (
        CorpusDatasetType,
        EncodeKwargs,
        QueryDatasetType,
        RetrievalOutputType,
        TopRankedDocumentsType,
    )

logger = logging.getLogger(__name__)


def _unicode_tokenize(text: str) -> list[str]:
    return list(unicodedata.normalize("NFKC", text).lower().replace(" ", ""))


# ISO 639-3 → (bm25s stopwords key, PyStemmer language name, custom tokenizer name).
# bm25s keys: en, de, nl, fr, es, pt, it, ru, sv, no, zh (or None = no stopwords).
# PyStemmer names: full English name as returned by Stemmer.algorithms() (or None = no stemmer).
# tokenizer: None = bm25s default, "char" = Unicode char split.
_ISO3_TO_LANG: dict[str, tuple[str | None, str | None, str | None]] = {
    # Both bm25s stopwords and PyStemmer
    "eng": ("en", "english", None),
    "deu": ("de", "german", None),
    "nld": ("nl", "dutch", None),
    "fra": ("fr", "french", None),
    "spa": ("es", "spanish", None),
    "por": ("pt", "portuguese", None),
    "ita": ("it", "italian", None),
    "rus": ("ru", "russian", None),
    "swe": ("sv", "swedish", None),
    "nob": ("no", "norwegian", None),  # Norwegian Bokmål
    "nno": ("no", "norwegian", None),  # Norwegian Nynorsk
    "nor": ("no", "norwegian", None),
    # Chinese: bm25s stopwords + char-level tokenizer
    "zho": ("zh", None, "char"),
    "cmn": ("zh", None, "char"),
    # No-space scripts without a dedicated tokenizer: character-level split
    "jpn": (None, None, "char"),  # Japanese (could add MeCab later)
    "tha": (None, None, "char"),  # Thai (could add pythainlp later)
    "khm": (None, None, "char"),  # Khmer
    "mya": (None, None, "char"),  # Myanmar
    "bod": (None, None, "char"),  # Tibetan
    "lao": (None, None, "char"),  # Lao
    # PyStemmer only (no bm25s stopword list — pass None to skip stopword removal)
    "ara": (None, "arabic", None),
    "hye": (None, "armenian", None),
    "eus": (None, "basque", None),
    "cat": (None, "catalan", None),
    "dan": (None, "danish", None),
    "epo": (None, "esperanto", None),
    "est": (None, "estonian", None),
    "fin": (None, "finnish", None),
    "ell": (None, "greek", None),
    "hin": (None, "hindi", None),
    "hun": (None, "hungarian", None),
    "ind": (None, "indonesian", None),
    "gle": (None, "irish", None),
    "lit": (None, "lithuanian", None),
    "nep": (None, "nepali", None),
    "ron": (None, "romanian", None),
    "srp": (None, "serbian", None),
    "tam": (None, "tamil", None),
    "tur": (None, "turkish", None),
    "yid": (None, "yiddish", None),
}


# Public-constant stopword lookup for bm25s keys (avoids importing the private helper).
def _get_stopwords(key: str) -> frozenset[str]:
    from bm25s import tokenization as _tok

    _map = {
        "en": _tok.STOPWORDS_EN,
        "de": _tok.STOPWORDS_GERMAN,
        "nl": _tok.STOPWORDS_DUTCH,
        "fr": _tok.STOPWORDS_FRENCH,
        "es": _tok.STOPWORDS_SPANISH,
        "pt": _tok.STOPWORDS_PORTUGUESE,
        "it": _tok.STOPWORDS_ITALIAN,
        "ru": _tok.STOPWORDS_RUSSIAN,
        "sv": _tok.STOPWORDS_SWEDISH,
        "no": _tok.STOPWORDS_NORWEGIAN,
        "zh": _tok.STOPWORDS_CHINESE,
    }
    return frozenset(_map[key])


def _get_language(task_metadata: TaskMetadata, hf_subset: str) -> str | None:
    """Return the ISO 639-3 language code for the task subset, or None if multilingual."""
    eval_langs = task_metadata.eval_langs
    langs = (
        eval_langs.get(hf_subset, []) if isinstance(eval_langs, dict) else eval_langs
    )
    iso3_codes = {lang.split("-")[0] for lang in langs}
    if len(iso3_codes) != 1:
        return None
    return next(iter(iso3_codes))


class BM25Tokenizer:
    """sklearn-style tokenizer for BM25 retrieval.

    Supports two backends selected via ``tokenizer``:

    * ``None`` — bm25s native whitespace tokeniser (+ optional Snowball stemmer).
    * ``str`` — named custom tokeniser: ``"char"``.
    * ``callable`` — any ``text -> list[str]`` function (e.g. a subword tokeniser).

    For languages without a named stopword list, tokens appearing in ≥
    ``freq_threshold`` of corpus documents are removed automatically
    (set ``freq_threshold=0`` to disable).  ``fit_transform`` is efficient: the
    bm25s path tokenises the corpus only once; the custom path applies the
    freq-stop computation in a single pass.

    ``transform`` reuses the stopword set learned during ``fit_transform``, so
    query tokenisation is consistent with the index.  bm25s reconciles
    query/corpus token IDs internally, so each ``transform`` call may build a
    fresh vocab without breaking retrieval.
    """

    def __init__(
        self,
        language: str | None,
        stopwords_key: str | None = None,
        stemmer_language: str | None = None,
        freq_threshold: float = 0.9,
        tokenizer: Callable[[str], list[str]] | None = None,
    ):
        # Resolve language defaults, then apply explicit overrides.
        # language=None means "no language assumptions" — no stopwords, stemmer, or tokenizer.
        if language is None:
            detected_sw, detected_stemmer, detected_tok = None, None, None
        else:
            detected_sw, detected_stemmer, detected_tok = _ISO3_TO_LANG[language]
        self.stopwords_key = stopwords_key if stopwords_key is not None else detected_sw
        stemmer_lang = (
            stemmer_language if stemmer_language is not None else detected_stemmer
        )
        # Explicit stopwords/stemmer override → skip language-specific tokenizer
        if stopwords_key is not None or stemmer_language is not None:
            # tokenizer kwarg still honoured if passed explicitly
            self._tok_arg = tokenizer
        else:
            self._tok_arg = tokenizer if tokenizer is not None else detected_tok

        self.freq_threshold = freq_threshold
        self._raw_tok = None  # resolved callable (custom path only)
        self._combined_stops: frozenset[str] = frozenset()
        self._combined_list: list[str] | None = None  # bm25s path cache
        if stemmer_lang:
            import Stemmer

            self.stemmer = Stemmer.Stemmer(stemmer_lang)
        else:
            self.stemmer = None

        logger.info(
            f"Language settings — stopwords: {self.stopwords_key!r}, "
            f"stemmer: {stemmer_lang!r}, tokenizer: {self._tok_arg!r}"
        )

    def fit(self, corpus_texts: list[str]) -> BM25Tokenizer:
        self.fit_transform(corpus_texts)
        return self

    def fit_transform(self, corpus_texts: list[str]):
        """Fit on corpus and return the encoded corpus as a ``Tokenized``."""
        if self._tok_arg is None:
            return self._fit_transform_bm25s(corpus_texts)
        self._raw_tok = (
            self._tok_arg if callable(self._tok_arg) else self._named_tok(self._tok_arg)
        )
        return self._fit_transform_custom(corpus_texts)

    def transform(self, texts: list[str]):
        """Tokenize ``texts`` using the stopword set learned during ``fit_transform``."""
        if self._tok_arg is None:
            return self._transform_bm25s(texts)
        return self._transform_custom(texts)

    def _fit_transform_bm25s(self, corpus_texts: list[str]):
        import bm25s
        from bm25s.tokenization import Tokenized

        # Fast path: named stopword list available, no freq-based stops needed.
        # Delegate entirely to bm25s to preserve its internal vocab/ID mapping.
        if self.stopwords_key is not None or self.freq_threshold == 0:
            named_stops = (
                _get_stopwords(self.stopwords_key)
                if self.stopwords_key
                else frozenset()
            )
            self._combined_stops = named_stops
            self._combined_list = list(named_stops) if named_stops else None
            return bm25s.tokenize(
                corpus_texts, stopwords=self._combined_list, stemmer=self.stemmer
            )

        # Slow path: no named stopword list — compute freq-based stops from corpus.
        raw = bm25s.tokenize(corpus_texts, stopwords=None, stemmer=self.stemmer)

        id_to_token = {v: k for k, v in raw.vocab.items()}
        n = len(corpus_texts)
        doc_freq: dict[str, int] = {}
        for doc_ids in raw.ids:
            for t in {id_to_token[i] for i in doc_ids}:
                doc_freq[t] = doc_freq.get(t, 0) + 1
        freq_stops = frozenset(
            t for t, df in doc_freq.items() if df / n >= self.freq_threshold
        )
        logger.info(
            f"Freq-stopwords: {len(freq_stops)} tokens removed (threshold={self.freq_threshold})"
        )

        self._combined_stops = freq_stops
        self._combined_list = list(freq_stops) if freq_stops else None

        stop_ids = frozenset(raw.vocab[t] for t in freq_stops if t in raw.vocab)
        filtered_ids = [
            [i for i in doc_ids if i not in stop_ids] for doc_ids in raw.ids
        ]
        return Tokenized(ids=filtered_ids, vocab=raw.vocab)

    def _transform_bm25s(self, texts: list[str]):
        import bm25s

        return bm25s.tokenize(
            texts, stopwords=self._combined_list, stemmer=self.stemmer
        )

    def _fit_transform_custom(self, corpus_texts: list[str]):
        raw_token_lists = [self._raw_tok(text) for text in corpus_texts]

        freq_stops: frozenset[str] = frozenset()
        if self.stopwords_key is None and self.freq_threshold > 0:
            n = len(corpus_texts)
            doc_freq: dict[str, int] = {}
            for tokens in raw_token_lists:
                for t in set(tokens):
                    doc_freq[t] = doc_freq.get(t, 0) + 1
            freq_stops = frozenset(
                t for t, df in doc_freq.items() if df / n >= self.freq_threshold
            )
            logger.info(
                f"Freq-stopwords: {len(freq_stops)} tokens removed (threshold={self.freq_threshold})"
            )

        named_stops = (
            _get_stopwords(self.stopwords_key) if self.stopwords_key else frozenset()
        )
        self._combined_stops = named_stops | freq_stops

        filtered = [
            [t for t in toks if t not in self._combined_stops]
            for toks in raw_token_lists
        ]
        return self._to_tokenized(filtered)

    def _transform_custom(self, texts: list[str]):
        token_lists = [
            [t for t in self._raw_tok(text) if t not in self._combined_stops]
            for text in texts
        ]
        return self._to_tokenized(token_lists)

    @staticmethod
    def _named_tok(name: str) -> Callable[[str], list[str]]:
        if name == "char":
            return _unicode_tokenize
        raise ValueError(f"Unknown tokenizer name: {name!r}")

    @staticmethod
    def _to_tokenized(token_lists: list[list[str]]):
        from bm25s.tokenization import Tokenized

        vocab: dict[str, int] = {}
        ids = []
        for tokens in token_lists:
            row = []
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
                row.append(vocab[t])
            ids.append(row)
        return Tokenized(ids=ids, vocab=vocab)


class BM25Search:
    """Language-aware BM25 retrieval model.

    At index time the task's ``eval_langs`` metadata is inspected to select the
    right stopword list, Snowball stemmer, and tokenizer automatically:

    * **Stopwords** — bm25s built-in lists for EN/DE/NL/FR/ES/PT/IT/RU/SV/NO/ZH.
    * **Stemmer** — PyStemmer (Snowball) for 30+ languages.
    * **Tokenizer** — character-level unigrams for logographic scripts (CJK,
      Japanese, Thai, …); default whitespace tokenisation for everything else.
    * **Freq-based stopwords** — for languages that have no named stopword list,
      tokens appearing in ≥ ``freq_threshold`` fraction of corpus documents are
      removed automatically (set ``freq_threshold=0`` to disable).

    Pass explicit ``stopwords`` or ``stemmer_language`` to pin those settings and
    skip auto-detection for the corresponding parameter.
    """

    def __init__(
        self,
        previous_results: str | None = None,
        stopwords: str | None = None,
        stemmer_language: str | None = None,
        freq_threshold: float = 0.9,
        tokenizer: str | Callable[[str], list[str]] | None = None,
        **kwargs,
    ):
        """
        Args:
            stopwords: bm25s stopwords key (e.g. ``"en"``, ``"zh"``). ``None`` auto-detects
                from task language at index time.
            stemmer_language: PyStemmer language name (e.g. ``"english"``). ``None`` auto-detects.
            freq_threshold: Remove tokens appearing in this fraction of corpus docs when no
                named stopword list is available. Set to ``0`` to disable.
            tokenizer: Custom tokenizer. A HuggingFace tokenizer name (str) or any
                ``text -> list[str]`` callable. When set, language-based stopwords and
                stemmer are skipped.
        """
        self.model = None
        self._stopwords_cfg = stopwords
        self._stemmer_cfg = stemmer_language
        self._freq_threshold = freq_threshold
        self._tokenizer_cfg = self._resolve_tokenizer(tokenizer)
        self._tokenizer = None
        self.retriever = None
        self.corpus_idx_to_id: dict[int, str] = {}

    @staticmethod
    def _resolve_tokenizer(
        tokenizer: str | Callable[[str], list[str]] | None,
    ) -> Callable[[str], list[str]] | None:
        if tokenizer is None or callable(tokenizer):
            return tokenizer
        from tokenizers import Tokenizer

        hf_tok = Tokenizer.from_pretrained(tokenizer)

        def _tok(text: str) -> list[str]:
            raw = hf_tok.encode(text, add_special_tokens=False).tokens
            return [
                t.replace(" ", "").replace("▁", "")
                for t in raw
                if t.replace(" ", "").replace("▁", "")
            ]

        return _tok

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None = None,
    ) -> None:
        import bm25s

        corpus_texts = [
            "\n".join([doc.get("title", ""), doc["text"]]) for doc in corpus
        ]
        # When a custom tokenizer callable is provided, skip language-based preprocessing.
        language = (
            None
            if self._tokenizer_cfg
            else (_get_language(task_metadata, hf_subset) or "eng")
        )
        self._tokenizer = BM25Tokenizer(
            language=language,
            stopwords_key=self._stopwords_cfg,
            stemmer_language=self._stemmer_cfg,
            freq_threshold=self._freq_threshold,
            tokenizer=self._tokenizer_cfg,
        )
        logger.info("Encoding Corpus...")
        encoded_corpus = self._tokenizer.fit_transform(corpus_texts)
        logger.info(
            f"Indexing Corpus... {len(encoded_corpus.ids):,} documents, {len(encoded_corpus.vocab):,} vocab"
        )
        self.retriever = bm25s.BM25()
        self.retriever.index(encoded_corpus)
        self.corpus_idx_to_id = {i: row["id"] for i, row in enumerate(corpus)}

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None = None,
        num_proc: int | None = None,
    ) -> RetrievalOutputType:
        if self._tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call `index` first.")
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call `index` first.")

        logger.info("Encoding Queries...")
        query_ids = list(queries["id"])
        results = {qid: {} for qid in query_ids}
        processed = _combine_queries_with_instruction_text(queries)
        queries_texts = list(processed["text"])
        query_token_strs = self._tokenizer.transform(queries_texts)

        logger.info(f"Retrieving Results... {len(queries):,} queries")

        queries_results, queries_scores = self.retriever.retrieve(
            query_token_strs,
            k=min(top_k, len(self.corpus_idx_to_id)),
        )

        # Iterate over queries
        for qi, qid in enumerate(query_ids):
            query_results = queries_results[qi]
            scores = queries_scores[qi]
            doc_id_to_score = {}
            query_documents = (
                top_ranked[qid] if top_ranked and qid in top_ranked else None
            )

            # Iterate over results
            for doc_idx, score in zip(query_results, scores):
                doc_id = self.corpus_idx_to_id[doc_idx]

                # handle reranking with a filtered set of documents
                if query_documents is not None and doc_id not in query_documents:
                    continue
                doc_id_to_score[doc_id] = float(score)

            results[qid] = doc_id_to_score

        return results


def bm25_loader(model_name, **kwargs) -> SearchProtocol:
    return BM25Search(**kwargs)


_BM25_CITATION = """@misc{bm25s,
      title={BM25S: Orders of magnitude faster lexical search via eager sparse scoring},
      author={Xing Han Lù},
      year={2024},
      eprint={2407.03618},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.03618},
}"""

qwen_languages = [
    "afr-Latn",
    "ara-Arab",
    "aze-Latn",
    "bel-Cyrl",
    "bul-Cyrl",
    "ben-Beng",
    "cat-Latn",
    "ceb-Latn",
    "ces-Latn",
    "cym-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "spa-Latn",
    "est-Latn",
    "eus-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "glg-Latn",
    "guj-Gujr",
    "heb-Hebr",
    "hin-Deva",
    "hrv-Latn",
    "hat-Latn",
    "hun-Latn",
    "hye-Armn",
    "ind-Latn",
    "isl-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "jav-Latn",
    "kat-Geor",
    "kaz-Cyrl",
    "khm-Khmr",
    "kan-Knda",
    "kor-Hang",
    "kir-Cyrl",
    "lao-Laoo",
    "lit-Latn",
    "lav-Latn",
    "mkd-Cyrl",
    "mal-Mlym",
    "mon-Cyrl",
    "mar-Deva",
    "msa-Latn",
    "mya-Mymr",
    "nep-Deva",
    "nld-Latn",
    "nor-Latn",
    "nob-Latn",
    "nno-Latn",
    "pan-Guru",
    "pol-Latn",
    "por-Latn",
    "que-Latn",
    "ron-Latn",
    "rus-Cyrl",
    "sin-Sinh",
    "slk-Latn",
    "slv-Latn",
    "swa-Latn",
    "tam-Taml",
    "tel-Telu",
    "tha-Thai",
    "tgl-Latn",
    "tur-Latn",
    "ukr-Cyrl",
    "urd-Arab",
    "vie-Latn",
    "yor-Latn",
    "zho-Hans",
]

# bm25 likely supports more languages, but these are the ones that has a tokenizer and/or stopword list available in bm25s or PyStemmer
bm25_s_languages = [
    # Latin-script languages with bm25s stopword lists
    "eng-Latn",
    "deu-Latn",
    "nld-Latn",
    "fra-Latn",
    "spa-Latn",
    "por-Latn",
    "ita-Latn",
    "swe-Latn",
    "nob-Latn",
    "nno-Latn",
    "nor-Latn",
    # Chinese (Jieba word segmentation + bm25s "zh" stopwords)
    "zho-Hans",
    "zho-Hant",
    "cmn-Hans",
    "cmn-Hant",
]

bm25_s = ModelMeta(
    loader=bm25_loader,
    extra_requirements_groups=["bm25s"],
    name="mteb/baseline-bm25s",
    model_type=["sparse"],
    languages=bm25_s_languages,
    open_weights=True,
    revision="0_3_0",
    release_date="2026-05-06",
    n_parameters=0,
    n_embedding_parameters=0,
    memory_usage_mb=None,
    embed_dim=None,
    license=None,
    max_tokens=None,
    reference="https://github.com/xhluca/bm25s",
    similarity_fn_name=None,
    framework=[],
    use_instructions=False,
    public_training_code="https://github.com/xhluca/bm25s",
    public_training_data=None,
    training_datasets=None,
    citation=_BM25_CITATION,
)

bm25_s_subword = ModelMeta(
    loader=bm25_loader,
    loader_kwargs={"tokenizer": "Qwen/Qwen3-0.6B"},
    extra_requirements_groups=["bm25s"],
    name="mteb/baseline-bm25s-subword",
    model_type=["sparse"],
    languages=qwen_languages,
    open_weights=True,
    revision="0_1_0",
    release_date="2026-05-06",
    n_parameters=0,
    n_embedding_parameters=0,
    memory_usage_mb=None,
    embed_dim=None,
    license=None,
    max_tokens=None,
    reference="https://github.com/xhluca/bm25s",
    similarity_fn_name=None,
    framework=[],
    use_instructions=False,
    public_training_code="https://github.com/xhluca/bm25s",
    public_training_data=None,
    training_datasets=None,
    citation=_BM25_CITATION,
)

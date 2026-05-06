from __future__ import annotations

import logging
import re
import unicodedata
from typing import TYPE_CHECKING

from mteb._create_dataloaders import _combine_queries_with_instruction_text
from mteb.models.model_meta import ModelMeta

if TYPE_CHECKING:
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

# Logographic/no-space scripts where each character is its own token.
# Includes CJK, Japanese kana, Hangul, Thai, Khmer, Myanmar.
_NO_SPACE_SCRIPT = (
    r"一-鿿"  # CJK Unified Ideographs
    r"㐀-䶿"  # CJK Extension A
    r"豈-﫿"  # CJK Compatibility Ideographs
    r"぀-ゟ"  # Hiragana
    r"゠-ヿ"  # Katakana
    r"가-힣"  # Hangul Syllables
    r"฀-๿"  # Thai
    r"ក-៿"  # Khmer
    r"က-႟"  # Myanmar
)
# Matches a single logographic character OR a sequence of non-logographic word characters.
_TOKEN_RE = re.compile(
    rf"[{_NO_SPACE_SCRIPT}]|[^{_NO_SPACE_SCRIPT}\W]+",
    re.UNICODE,
)


def _unicode_tokenize(text: str) -> list[str]:
    """Language-agnostic tokenizer using Unicode script detection.

    Yields character unigrams for logographic scripts (CJK, Thai, etc.) and
    whitespace-split words for Latin/Cyrillic/Arabic/etc.  No language
    knowledge or external models required.
    """
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower())


# ISO 639-3 → (bm25s stopwords key, PyStemmer language name, custom tokenizer name).
# bm25s keys: en, de, nl, fr, es, pt, it, ru, sv, no, zh (or None = no stopwords).
# PyStemmer names: full English name as returned by Stemmer.algorithms() (or None = no stemmer).
# tokenizer: None = bm25s default, "jieba" = Jieba word segmenter, "char" = Unicode char split.
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
    # Chinese: bm25s stopwords + Jieba word segmenter (falls back to char-level if unavailable)
    "zho": ("zh", None, "jieba"),
    "cmn": ("zh", None, "jieba"),
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

_DEFAULT_LANG: tuple[str | None, str | None, str | None] = ("en", "english", None)


def _resolve_language(
    task_metadata: TaskMetadata, hf_subset: str
) -> tuple[str | None, str | None, str | None]:
    """Return (bm25s_stopwords_key, pystemmer_lang, tokenizer_name) for the task's language.

    Falls back to English defaults when the task covers multiple languages or
    the language is not in the supported mapping.
    """
    eval_langs = task_metadata.eval_langs
    langs: list[str] = (
        eval_langs.get(hf_subset, []) if isinstance(eval_langs, dict) else eval_langs
    )
    iso3_codes = {lang.split("-")[0] for lang in langs}

    if len(iso3_codes) != 1:
        return _DEFAULT_LANG

    iso3 = next(iter(iso3_codes))
    return _ISO3_TO_LANG.get(iso3, _DEFAULT_LANG)


def _make_tokenizer_fn(name: str | None):
    """Return a callable ``text -> list[str]``, or None for the bm25s default."""
    if name is None:
        return None
    if name == "char":
        return _unicode_tokenize
    if name == "jieba":
        try:
            import jieba

            def _jieba_tok(text: str) -> list[str]:
                return [t for t in jieba.lcut(text) if t.strip()]

            return _jieba_tok
        except ImportError:
            logger.warning(
                "jieba not installed — falling back to character-level tokenization. "
                "Install with: pip install jieba"
            )
            return _unicode_tokenize
    raise ValueError(f"Unknown tokenizer name: {name!r}")


class BM25Search:
    """Language-aware BM25 retrieval model.

    At index time the task's ``eval_langs`` metadata is inspected to select the
    right stopword list, Snowball stemmer, and tokenizer automatically:

    * **Stopwords** — bm25s built-in lists for EN/DE/NL/FR/ES/PT/IT/RU/SV/NO/ZH.
    * **Stemmer** — PyStemmer (Snowball) for 30+ languages.
    * **Tokenizer** — Jieba word segmentation for Chinese; character-level
      unigrams for other logographic scripts (Japanese, Thai, …); default
      whitespace tokenisation for everything else.
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
        **kwargs,
    ):
        self.model = None
        self._stopwords_cfg = stopwords  # None → auto-detect in index()
        self._stemmer_cfg = stemmer_language  # None → auto-detect in index()
        self._freq_threshold = freq_threshold
        self.stopwords: str | None = None
        self.stemmer = None
        self._tokenizer_fn = None
        self._freq_stopwords: frozenset[str] = frozenset()
        self._corpus_vocab: dict[str, int] = {}
        self.retriever = None
        self.corpus_idx_to_id: dict[int, str] = {}

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
        detected_stopwords, detected_stemmer, tokenizer_name = _resolve_language(
            task_metadata, hf_subset
        )
        # Use explicit user values where provided; fall back to detected values.
        self.stopwords = (
            self._stopwords_cfg
            if self._stopwords_cfg is not None
            else detected_stopwords
        )
        stemmer_lang = (
            self._stemmer_cfg if self._stemmer_cfg is not None else detected_stemmer
        )
        # Tokenizer is always derived from language detection (no explicit override yet).
        if self._stopwords_cfg is not None or self._stemmer_cfg is not None:
            tokenizer_name = None  # explicit config → don't inject a custom tokenizer

        if stemmer_lang:
            import Stemmer

            self.stemmer = Stemmer.Stemmer(stemmer_lang)
        else:
            self.stemmer = None
        self._tokenizer_fn = _make_tokenizer_fn(tokenizer_name)
        self._corpus_vocab = {}  # reset for this corpus
        self._freq_stopwords = frozenset()
        logger.info(
            f"Language settings — stopwords: {self.stopwords!r}, "
            f"stemmer: {stemmer_lang!r}, tokenizer: {tokenizer_name!r}"
        )

        logger.info("Encoding Corpus...")
        corpus_texts = [
            "\n".join([doc.get("title", ""), doc["text"]]) for doc in corpus
        ]  # concatenate all document values (title, text, ...)

        if self.stopwords is None and self._freq_threshold > 0:
            self._freq_stopwords = self._build_freq_stopwords(corpus_texts)

        encoded_corpus = self._encode(corpus_texts)

        logger.info(
            f"Indexing Corpus... {len(encoded_corpus.ids):,} documents, {len(encoded_corpus.vocab):,} vocab"
        )

        # Create the BM25 model and index the corpus
        import bm25s

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
        logger.info("Encoding Queries...")
        query_ids = list(queries["id"])
        results = {qid: {} for qid in query_ids}
        processed = _combine_queries_with_instruction_text(queries)
        queries_texts = processed["text"]
        query_token_strs = self._encode(queries_texts)

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

    def _build_freq_stopwords(self, corpus_texts: list[str]) -> frozenset[str]:
        """Return tokens appearing in >= freq_threshold fraction of corpus docs."""
        n = len(corpus_texts)
        if self._tokenizer_fn is not None:
            token_lists: list[list[str]] = [self._tokenizer_fn(t) for t in corpus_texts]
        else:
            import bm25s

            tokenized = bm25s.tokenize(
                corpus_texts, stopwords=None, stemmer=self.stemmer
            )
            id_to_token = {v: k for k, v in tokenized.vocab.items()}
            token_lists = [[id_to_token[i] for i in doc] for doc in tokenized.ids]

        doc_freq: dict[str, int] = {}
        for tokens in token_lists:
            for t in set(tokens):
                doc_freq[t] = doc_freq.get(t, 0) + 1

        stops = frozenset(
            t for t, df in doc_freq.items() if df / n >= self._freq_threshold
        )
        logger.info(
            f"Freq-stopwords: {len(stops)} tokens removed (threshold={self._freq_threshold})"
        )
        return stops

    def _encode(self, texts: list[str]):
        """Tokenize texts using bm25s. Not to be confused with EncoderProtocol.encode()."""
        import bm25s

        if self._tokenizer_fn is None:
            if self._freq_stopwords:
                # Merge named stopwords with corpus-frequency stopwords.
                # bm25s applies the stemmer before checking, so freq_stopwords (already
                # stemmed from the pre-pass) will match correctly.
                from bm25s.tokenization import _infer_stopwords

                named = list(_infer_stopwords(self.stopwords)) if self.stopwords else []
                combined = named + list(self._freq_stopwords)
                return bm25s.tokenize(texts, stopwords=combined, stemmer=self.stemmer)
            return bm25s.tokenize(texts, stopwords=self.stopwords, stemmer=self.stemmer)

        # Custom tokenizer path — build Tokenized manually and persist corpus vocab
        # so query token IDs match the corpus index.
        from bm25s.tokenization import Tokenized, _infer_stopwords

        stopwords_set = (
            frozenset(_infer_stopwords(self.stopwords))
            if self.stopwords
            else frozenset()
        ) | self._freq_stopwords
        token_lists = [
            [t for t in self._tokenizer_fn(text) if t not in stopwords_set]
            for text in texts
        ]

        if self._corpus_vocab:
            vocab = self._corpus_vocab
            encoded_ids = [
                [vocab[t] for t in tokens if t in vocab] for tokens in token_lists
            ]
        else:
            vocab = {}
            encoded_ids = []
            for tokens in token_lists:
                ids = []
                for t in tokens:
                    if t not in vocab:
                        vocab[t] = len(vocab)
                    ids.append(vocab[t])
                encoded_ids.append(ids)
            self._corpus_vocab = vocab

        return Tokenized(ids=encoded_ids, vocab=vocab)


class BM25SubwordSearch(BM25Search):
    """BM25 search using a HuggingFace subword tokenizer for multilingual support.

    Unlike the standard BM25 model that relies on whitespace splitting and
    PyStemmer, this uses a trained subword tokenizer (default: Qwen/Qwen3-0.6B)
    that handles non-Latin scripts without requiring language-specific knowledge.
    """

    def __init__(
        self,
        previous_results: str | None = None,
        tokenizer_name: str = "Qwen/Qwen3-0.6B",
        **kwargs,
    ):
        from tokenizers import Tokenizer

        self.model = None
        self._stopwords_cfg = None
        self._stemmer_cfg = None
        self._freq_threshold = 0.0
        self.stopwords = None
        self.stemmer = None
        self._tokenizer_fn = None
        self._freq_stopwords: frozenset[str] = frozenset()
        self.retriever = None
        self.corpus_idx_to_id = {}
        self._corpus_vocab: dict[str, int] = {}
        self.hf_tokenizer = Tokenizer.from_pretrained(tokenizer_name)

    def _tokenize_raw(self, texts: list[str]) -> list[list[str]]:
        token_lists = []
        for text in texts:
            raw_tokens = self.hf_tokenizer.encode(text, add_special_tokens=False).tokens
            clean = [
                t.replace(" ", "").replace("\u2581", "")
                for t in raw_tokens
                if t.replace(" ", "").replace("\u2581", "")
            ]
            token_lists.append(clean)
        return token_lists

    def _encode(self, texts: list[str]):
        """Tokenize texts using a HuggingFace subword tokenizer, then wrap for bm25s."""
        from bm25s.tokenization import Tokenized

        token_lists = self._tokenize_raw(texts)

        if self._corpus_vocab:
            # Query encoding: reuse corpus vocab so token IDs match the index.
            vocab = self._corpus_vocab
            encoded_ids = [
                [vocab[t] for t in tokens if t in vocab] for tokens in token_lists
            ]
        else:
            # Corpus encoding: build vocab and persist it for later query calls.
            vocab = {}
            encoded_ids = []
            for tokens in token_lists:
                ids = []
                for t in tokens:
                    if t not in vocab:
                        vocab[t] = len(vocab)
                    ids.append(vocab[t])
                encoded_ids.append(ids)
            self._corpus_vocab = vocab

        return Tokenized(ids=encoded_ids, vocab=vocab)


def bm25_loader(model_name, **kwargs) -> SearchProtocol:
    return BM25Search(**kwargs)


def bm25_subword_loader(model_name, **kwargs) -> SearchProtocol:
    return BM25SubwordSearch(**kwargs)


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
    model_type=["dense"],
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
    loader=bm25_subword_loader,
    extra_requirements_groups=["bm25s"],
    name="mteb/baseline-bm25s-subword",
    model_type=["dense"],
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

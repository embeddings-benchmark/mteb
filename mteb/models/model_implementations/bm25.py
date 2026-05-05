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

_LANG_AUTO = "auto"  # sentinel: derive language from task_metadata at index time

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
    """BM25 search using PyStemmer for stemming and bm25s for indexing.

    By default the stemmer and stopwords are derived automatically from the
    task metadata at index time.  Pass explicit ``stopwords`` / ``stemmer_language``
    values to override.
    """

    def __init__(
        self,
        previous_results: str | None = None,
        stopwords: str | None = _LANG_AUTO,
        stemmer_language: str | None = _LANG_AUTO,
        **kwargs,
    ):
        self.model = None
        self._stopwords_cfg = stopwords
        self._stemmer_cfg = stemmer_language
        self.stopwords: str | None = "en"        # overwritten in index()
        self.stemmer = None                       # overwritten in index()
        self._tokenizer_fn = None                 # overwritten in index()
        self._corpus_vocab: dict[str, int] = {}  # persisted for query encoding
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
        import Stemmer

        stopwords_cfg = self._stopwords_cfg
        stemmer_cfg = self._stemmer_cfg
        if _LANG_AUTO in (stopwords_cfg, stemmer_cfg):
            detected_stopwords, detected_stemmer, tokenizer_name = _resolve_language(
                task_metadata, hf_subset
            )
            self.stopwords = (
                detected_stopwords if stopwords_cfg == _LANG_AUTO else stopwords_cfg
            )
            stemmer_lang = (
                detected_stemmer if stemmer_cfg == _LANG_AUTO else stemmer_cfg
            )
        else:
            self.stopwords = stopwords_cfg
            stemmer_lang = stemmer_cfg
            tokenizer_name = None

        self.stemmer = Stemmer.Stemmer(stemmer_lang) if stemmer_lang else None
        self._tokenizer_fn = _make_tokenizer_fn(tokenizer_name)
        self._corpus_vocab = {}  # reset for this corpus
        logger.info(
            f"Language settings — stopwords: {self.stopwords!r}, "
            f"stemmer: {stemmer_lang!r}, tokenizer: {tokenizer_name!r}"
        )

        logger.info("Encoding Corpus...")
        corpus_texts = [
            "\n".join([doc.get("title", ""), doc["text"]]) for doc in corpus
        ]  # concatenate all document values (title, text, ...)
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

    def _encode(self, texts: list[str]):
        """Tokenize texts using bm25s. Not to be confused with EncoderProtocol.encode()."""
        import bm25s

        if self._tokenizer_fn is None:
            return bm25s.tokenize(texts, stopwords=self.stopwords, stemmer=self.stemmer)

        # Custom tokenizer path — build Tokenized manually and persist corpus vocab
        # so query token IDs match the corpus index.
        from bm25s.tokenization import Tokenized, _infer_stopwords

        stopwords_set = frozenset(_infer_stopwords(self.stopwords) if self.stopwords else [])
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


class BM25MultilingualSearch(BM25Search):
    """BM25 search using a HuggingFace subword tokenizer for multilingual support.

    Unlike the standard BM25 model that relies on whitespace splitting and
    PyStemmer, this uses a trained multilingual subword tokenizer (default:
    xlm-roberta-base) that handles non-Latin scripts (Chinese, Japanese, etc.)
    without requiring language-specific stemmers.
    """

    def __init__(
        self,
        previous_results: str | None = None,
        tokenizer_name: str = "xlm-roberta-base",
        **kwargs,
    ):
        from tokenizers import Tokenizer

        self.model = None
        self.stopwords = None
        self.stemmer = None
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


class BM25UnicodeSplitSearch(BM25Search):
    """BM25 with script-aware Unicode tokenization; no language knowledge needed.

    Uses character unigrams for logographic scripts (CJK, Thai, Khmer, etc.)
    and whitespace-split words for Latin/Cyrillic/Arabic/Hebrew/etc., with
    NFKC normalisation and lowercasing.  No external models or language
    detection required.
    """

    def __init__(self, previous_results: str | None = None, **kwargs):
        self.model = None
        self.stemmer = None
        self.stopwords = None
        self.retriever = None
        self.corpus_idx_to_id: dict[int, str] = {}
        self._corpus_vocab: dict[str, int] = {}

    def index(self, corpus, *, task_metadata, hf_split, hf_subset, encode_kwargs, num_proc=None):
        self._corpus_vocab = {}  # reset so _encode rebuilds vocab from this corpus
        super().index(
            corpus,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            encode_kwargs=encode_kwargs,
            num_proc=num_proc,
        )

    def _encode(self, texts: list[str]):
        from bm25s.tokenization import Tokenized

        token_lists = [_unicode_tokenize(t) for t in texts]

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
    import bm25s
    import Stemmer

    class BM25Search:
        """BM25 search"""

        retriever: bm25s.BM25
        corpus_idx_to_id: dict[int, str]

        def __init__(
            self,
            previous_results: str | None = None,
            stopwords: str = "en",
            stemmer_language: str | None = "english",
            **kwargs,
        ):
            self.model = None

            self.stopwords = stopwords
            self.stemmer = (
                Stemmer.Stemmer(stemmer_language) if stemmer_language else None
            )

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
            logger.info("Encoding Corpus...")
            corpus_texts = [
                "\n".join([doc.get("title", ""), doc["text"]]) for doc in corpus
            ]  # concatenate all document values (title, text, ...)
            encoded_corpus = self._encode(corpus_texts)

            logger.info(
                f"Indexing Corpus... {len(encoded_corpus.ids):,} documents, {len(encoded_corpus.vocab):,} vocab"
            )

            # Create the BM25 model and index the corpus
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

        def _encode(self, texts: list[str]):
            """Tokenize texts using bm25s. Not to be confused with EncoderProtocol.encode()."""
            return bm25s.tokenize(texts, stopwords=self.stopwords, stemmer=self.stemmer)

    return BM25Search(**kwargs)


def bm25_multilingual_loader(model_name, **kwargs) -> SearchProtocol:
    return BM25MultilingualSearch(**kwargs)


def bm25_unicode_loader(model_name, **kwargs) -> SearchProtocol:
    return BM25UnicodeSplitSearch(**kwargs)


def bm25_lang_aware_loader(model_name, **kwargs) -> SearchProtocol:
    return BM25Search(**kwargs)


bm25_s = ModelMeta(
    loader=bm25_loader,
    extra_requirements_groups=["bm25s"],
    name="mteb/baseline-bm25s",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="0_1_10",
    release_date="2024-07-10",  # release of version 0.1.10
    n_parameters=None,
    n_embedding_parameters=None,
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
    citation="""@misc{bm25s,
      title={BM25S: Orders of magnitude faster lexical search via eager sparse scoring},
      author={Xing Han Lù},
      year={2024},
      eprint={2407.03618},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.03618},
}""",
)

_BM25_CITATION = """@misc{bm25s,
      title={BM25S: Orders of magnitude faster lexical search via eager sparse scoring},
      author={Xing Han Lù},
      year={2024},
      eprint={2407.03618},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.03618},
}"""

bm25_s_multilingual = ModelMeta(
    loader=bm25_multilingual_loader,
    extra_requirements_groups=["bm25s"],
    name="mteb/baseline-bm25s-multilingual",
    model_type=["dense"],
    languages=None,
    open_weights=True,
    revision="0_1_0",
    release_date="2026-04-15",
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

bm25_s_unicode = ModelMeta(
    loader=bm25_unicode_loader,
    extra_requirements_groups=["bm25s"],
    name="mteb/baseline-bm25s-unicode",
    model_type=["dense"],
    languages=None,
    open_weights=True,
    revision="0_1_0",
    release_date="2026-05-05",
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

bm25_s_lang_aware = ModelMeta(
    loader=bm25_lang_aware_loader,
    extra_requirements_groups=["bm25s"],
    name="mteb/baseline-bm25s-lang-aware",
    model_type=["dense"],
    languages=None,
    open_weights=True,
    revision="0_1_0",
    release_date="2026-05-05",
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

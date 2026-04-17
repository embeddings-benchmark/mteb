from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mteb._create_dataloaders import _combine_queries_with_instruction_text
from mteb._requires_package import requires_package
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


class BM25Search:
    """BM25 search using PyStemmer for English/Latin-script languages."""

    def __init__(
        self,
        previous_results: str | None = None,
        stopwords: str = "en",
        stemmer_language: str | None = "english",
        **kwargs,
    ):
        requires_package(self, "bm25s", "bm25", "pip install 'mteb[bm25s]'")
        import Stemmer

        self.model = None
        self.stopwords = stopwords
        self.stemmer = Stemmer.Stemmer(stemmer_language) if stemmer_language else None
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
        processed = queries.map(
            _combine_queries_with_instruction_text,
            desc="Processing queries for dataloading",
        )
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

        return bm25s.tokenize(texts, stopwords=self.stopwords, stemmer=self.stemmer)


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
        requires_package(
            self, "bm25s", "bm25-multilingual", "pip install 'mteb[bm25s]'"
        )
        from tokenizers import Tokenizer

        self.model = None
        self.stopwords = None
        self.stemmer = None
        self.retriever = None
        self.corpus_idx_to_id = {}
        self.hf_tokenizer = Tokenizer.from_pretrained(tokenizer_name)

    def _encode(self, texts: list[str]):
        """Tokenize texts using a HuggingFace subword tokenizer, then wrap for bm25s."""
        from bm25s.tokenization import Tokenized

        token_lists = []
        for text in texts:
            raw_tokens = self.hf_tokenizer.encode(text, add_special_tokens=False).tokens
            clean = [
                t.replace(" ", "").replace("\u2581", "")
                for t in raw_tokens
                if t.replace(" ", "").replace("\u2581", "")
            ]
            token_lists.append(clean)

        # Build corpus_tokens list and vocab mapping for bm25s
        vocab: dict[str, int] = {}
        encoded_ids = []
        for tokens in token_lists:
            ids = []
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
                ids.append(vocab[t])
            encoded_ids.append(ids)

        return Tokenized(ids=encoded_ids, vocab=vocab)


def bm25_loader(model_name, **kwargs) -> SearchProtocol:
    return BM25Search(**kwargs)


def bm25_multilingual_loader(model_name, **kwargs) -> SearchProtocol:
    return BM25MultilingualSearch(**kwargs)


bm25_s = ModelMeta(
    loader=bm25_loader,
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

bm25_s_multilingual = ModelMeta(
    loader=bm25_multilingual_loader,
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

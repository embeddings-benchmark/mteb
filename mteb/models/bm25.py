from __future__ import annotations

import logging
from functools import partial
from typing import Any, List, Optional, Union

from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel
from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts

logger = logging.getLogger(__name__)


def bm25_loader(**kwargs):
    try:
        import bm25s
        import Stemmer
    except ImportError:
        raise ImportError(
            "bm25s or Stemmer is not installed. Please install it with `pip install bm25s Stemmer`."
        )

    class BM25Search(DRESModel):
        """BM25 search"""

        def __init__(
            self,
            previous_results: str = None,
            stopwords: str = "en",
            stemmer_language: Optional[str] = "english",
            **kwargs,
        ):
            super().__init__(
                model=None,
                batch_size=1,
                corpus_chunk_size=1,
                previous_results=previous_results,
                **kwargs,
            )

            self.stopwords = stopwords
            self.stemmer = (
                Stemmer.Stemmer(stemmer_language) if stemmer_language else None
            )

        @classmethod
        def name(self):
            return "bm25s"

        def search(
            self,
            corpus: dict[str, dict[str, str]],
            queries: dict[str, Union[str, List[str]]],
            top_k: int,
            score_function: str,
            return_sorted: bool = False,
            **kwargs,
        ) -> dict[str, dict[str, float]]:
            logger.info("Encoding Corpus...")
            corpus_ids = list(corpus.keys())
            corpus_with_ids = [{"doc_id": cid, **corpus[cid]} for cid in corpus_ids]

            corpus_texts = [
                "\n".join([doc.get("title", ""), doc["text"]])
                for doc in corpus_with_ids
            ]  # concatenate all document values (title, text, ...)
            encoded_corpus = self.encode(corpus_texts)

            logger.info(
                f"Indexing Corpus... {len(encoded_corpus.ids):,} documents, {len(encoded_corpus.vocab):,} vocab"
            )

            # Create the BM25 model and index the corpus
            retriever = bm25s.BM25()
            retriever.index(encoded_corpus)

            logger.info("Encoding Queries...")
            query_ids = list(queries.keys())
            self.results = {qid: {} for qid in query_ids}
            queries_texts = [queries[qid] for qid in queries]

            query_token_strs = self.encode(queries_texts, return_ids=False)

            logger.info(f"Retrieving Results... {len(queries):,} queries")

            queries_results, queries_scores = retriever.retrieve(
                query_token_strs, corpus=corpus_with_ids, k=top_k
            )

            # Iterate over queries
            for qi, qid in enumerate(query_ids):
                doc_id_to_score = {}
                query_results = queries_results[qi]
                scores = queries_scores[qi]
                doc_id_to_score = {}

                # Iterate over results
                for ri in range(len(query_results)):
                    doc = query_results[ri]
                    score = scores[ri]
                    doc_id = doc["doc_id"]

                    doc_id_to_score[doc_id] = float(score)

                self.results[qid] = doc_id_to_score

            return self.results

        def encode(self, texts: List[str], **kwargs):
            """Encode input text as term vectors"""
            return bm25s.tokenize(texts, stopwords=self.stopwords, stemmer=self.stemmer)

        def encode_queries(
            self,
            queries: list[str],
            batch_size: int = 32,
            **kwargs: Any,
        ):
            return self.encode(queries, kwargs=kwargs)

        def encode_corpus(
            self,
            corpus: list[dict[str, str]] | dict[str, list[str]],
            batch_size: int = 32,
            **kwargs: Any,
        ):
            sentences = corpus_to_texts(corpus)
            return self.encode(sentences, kwargs=kwargs)

    return BM25Search(**kwargs)


bm25_s = ModelMeta(
    loader=partial(bm25_loader, model_name="bm25s"),  # type: ignore
    name="bm25s",
    languages=["eng_Latn"],
    open_source=True,
    revision="0_1_10",
    release_date="2024-07-10",  ## release of version 0.1.10
)

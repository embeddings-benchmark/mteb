from __future__ import annotations

import logging
from typing import Literal

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.requires_package import requires_package
from mteb.types import CorpusDatasetType, QueryDatasetType

logger = logging.getLogger(__name__)


def bm25_loader(model_name, **kwargs):
    requires_package(bm25_loader, "bm25s", model_name, "pip install mteb[bm25s]")
    import bm25s
    import Stemmer

    class BM25Search(AbsEncoder):
        """BM25 search"""

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

        @classmethod
        def name(cls) -> Literal["bm25s"]:
            return "bm25s"

        def search(
            self,
            corpus: CorpusDatasetType,
            queries: QueryDatasetType,
            top_k: int,
            **kwargs,
        ) -> dict[str, dict[str, float]]:
            logger.info("Encoding Corpus...")
            corpus_texts = [
                "\n".join([doc.get("title", ""), doc["text"]]) for doc in corpus
            ]  # concatenate all document values (title, text, ...)
            encoded_corpus = self.encode(corpus_texts)

            logger.info(
                f"Indexing Corpus... {len(encoded_corpus.ids):,} documents, {len(encoded_corpus.vocab):,} vocab"
            )

            # Create the BM25 model and index the corpus
            retriever = bm25s.BM25()
            retriever.index(encoded_corpus)

            logger.info("Encoding Queries...")
            query_ids = list(queries["id"])
            self.results = {qid: {} for qid in query_ids}
            queries_texts = queries["text"]

            query_token_strs = self.encode(queries_texts, return_ids=False)

            logger.info(f"Retrieving Results... {len(queries):,} queries")

            queries_results, queries_scores = retriever.retrieve(
                query_token_strs, corpus=corpus.to_list(), k=top_k
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
                    doc_id = doc["id"]

                    doc_id_to_score[doc_id] = float(score)

                self.results[qid] = doc_id_to_score

            return self.results

        def encode(self, texts: list[str], **kwargs):
            """Encode input text as term vectors"""
            return bm25s.tokenize(texts, stopwords=self.stopwords, stemmer=self.stemmer)  # type: ignore

    return BM25Search(**kwargs)


bm25_s = ModelMeta(
    loader=bm25_loader,
    name="bm25s",
    languages=["eng-Latn"],
    open_weights=True,
    revision="0_1_10",
    release_date="2024-07-10",  ## release of version 0.1.10
    n_parameters=None,
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
)

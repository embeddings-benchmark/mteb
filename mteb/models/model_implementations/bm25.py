import logging
from typing import Any

from mteb._create_dataloaders import _create_text_queries_dataloader
from mteb._requires_package import requires_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta
from mteb.models.models_protocols import SearchProtocol
from mteb.types import (
    CorpusDatasetType,
    InstructionDatasetType,
    QueryDatasetType,
    RetrievalOutputType,
    TopRankedDocumentsType,
)

logger = logging.getLogger(__name__)


def bm25_loader(model_name, **kwargs) -> SearchProtocol:
    requires_package(bm25_loader, "bm25s", model_name, "pip install mteb[bm25s]")
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
            encode_kwargs: dict[str, Any],
        ) -> None:
            logger.info("Encoding Corpus...")
            corpus_texts = [
                "\n".join([doc.get("title", ""), doc["text"]]) for doc in corpus
            ]  # concatenate all document values (title, text, ...)
            encoded_corpus = self.encode(corpus_texts)

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
            encode_kwargs: dict[str, Any],
            instructions: InstructionDatasetType | None = None,
            top_ranked: TopRankedDocumentsType | None = None,
        ) -> RetrievalOutputType:
            logger.info("Encoding Queries...")
            query_ids = list(queries["id"])
            results = {qid: {} for qid in query_ids}
            queries_loader = _create_text_queries_dataloader(queries)
            queries_texts = [text for batch in queries_loader for text in batch["text"]]

            query_token_strs = self.encode(queries_texts)

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

                # Iterate over results
                for ri in range(len(query_results)):
                    doc_idx = query_results[ri]
                    score = scores[ri]
                    doc_id = self.corpus_idx_to_id[doc_idx]

                    doc_id_to_score[doc_id] = float(score)

                results[qid] = doc_id_to_score

            return results

        def encode(self, texts: list[str]):
            """Encode input text as term vectors"""
            return bm25s.tokenize(texts, stopwords=self.stopwords, stemmer=self.stemmer)

    return BM25Search(**kwargs)


bm25_s = ModelMeta(
    loader=bm25_loader,
    name="bm25s",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="0_1_10",
    release_date="2024-07-10",  # release of version 0.1.10
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

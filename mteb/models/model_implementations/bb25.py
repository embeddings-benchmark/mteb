from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from mteb._create_dataloaders import _create_text_queries_dataloader
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


def _composite_prior(
    query_tfs: np.ndarray,
    doc_lengths: np.ndarray,
    avg_dl: float,
) -> np.ndarray:
    """Composite prior from Bayesian BM25 paper Section 4.2.

    Combines term frequency prior (Def 4.2.1) and field norm prior
    (Def 4.2.2) to produce document-level prior probabilities.

    Args:
        query_tfs: Total query term frequency per candidate document.
        doc_lengths: Token count per candidate document.
        avg_dl: Average document length across the corpus.

    Returns:
        Prior probabilities clipped to [0.1, 0.9].
    """
    p_tf = 0.2 + 0.7 * np.minimum(1.0, query_tfs / 10.0)

    if avg_dl > 0:
        norm_ratio = doc_lengths / avg_dl
    else:
        norm_ratio = np.ones_like(doc_lengths, dtype=np.float64)
    p_norm = 0.3 + 0.6 * (1.0 - np.minimum(1.0, np.abs(norm_ratio - 1.0) * 0.5))

    prior = 0.7 * p_tf + 0.3 * p_norm
    return np.clip(prior, 0.1, 0.9)


def bb25_loader(model_name, **kwargs) -> SearchProtocol:
    requires_package(bb25_loader, "bm25s", model_name, "pip install mteb[bm25s]")
    import bm25s
    import Stemmer

    class BB25Search:
        """Bayesian BM25 search using bm25s as the BM25 backend.

        Bayesian BM25 transforms traditional BM25 scores into calibrated
        probability estimates in [0, 1] through Bayesian inference with a
        sigmoid likelihood model and composite prior design.

        With the default prior_weight=0.0, the prior is a flat 0.5 for
        all documents. Since the sigmoid likelihood is strictly monotonic,
        this preserves BM25 rankings exactly while outputting calibrated
        probabilities suitable for hybrid search score fusion.

        Setting prior_weight > 0 enables the Composite Prior (Section 4.2
        of the paper), which re-adjusts rankings based on document-level
        evidence from query term frequency and document length signals.

        Architecture:
            1. bm25s handles fast BM25 indexing and top-k retrieval.
            2. Retrieved candidates are re-scored with Bayesian posterior:
               - Dynamic beta = median(BM25 scores) per query
               - Sigmoid likelihood: sigma(alpha * (score - beta))
               - prior = 0.5 + prior_weight * (composite_prior - 0.5)
               - Posterior via Bayes' rule
        """

        retriever: bm25s.BM25
        corpus_idx_to_id: dict[int, str]

        def __init__(
            self,
            previous_results: str | None = None,
            stopwords: str = "en",
            stemmer_language: str | None = "english",
            k1: float = 1.5,
            b: float = 0.75,
            alpha: float = 1.0,
            prior_weight: float = 0.0,
            **kwargs,
        ):
            self.k1 = k1
            self.b = b
            self.alpha = alpha
            self.prior_weight = prior_weight
            self.stopwords = stopwords
            self.stemmer = (
                Stemmer.Stemmer(stemmer_language) if stemmer_language else None
            )

        def _encode(self, texts: list[str]):
            """Tokenize texts using bm25s. Not to be confused with EncoderProtocol.encode()."""
            return bm25s.tokenize(texts, stopwords=self.stopwords, stemmer=self.stemmer)

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
            ]
            encoded_corpus = self._encode(corpus_texts)

            logger.info(
                f"Indexing Corpus... {len(encoded_corpus.ids):,} documents, "
                f"{len(encoded_corpus.vocab):,} vocab"
            )

            self.retriever = bm25s.BM25(k1=self.k1, b=self.b)
            self.retriever.index(encoded_corpus)
            self.corpus_idx_to_id = {i: row["id"] for i, row in enumerate(corpus)}

            if self.prior_weight > 0:
                # Per-document token IDs for composite prior computation.
                # Stored as compact numpy int32 arrays to minimize memory.
                self.corpus_token_ids = [
                    np.array(doc_ids, dtype=np.int32) for doc_ids in encoded_corpus.ids
                ]
                self.corpus_vocab = dict(encoded_corpus.vocab)
                self.doc_lengths = np.array(
                    [len(ids) for ids in encoded_corpus.ids], dtype=np.float64
                )
                self.avg_dl = (
                    float(self.doc_lengths.mean()) if len(self.doc_lengths) > 0 else 0.0
                )

            logger.info(f"Indexed {len(self.corpus_idx_to_id):,} documents")

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
            results: RetrievalOutputType = {qid: {} for qid in query_ids}
            queries_loader = _create_text_queries_dataloader(queries)
            queries_texts = [text for batch in queries_loader for text in batch["text"]]

            query_tokenized = self._encode(queries_texts)

            logger.info(f"Retrieving Results... {len(queries):,} queries")

            queries_results, queries_scores = self.retriever.retrieve(
                query_tokenized,
                k=min(top_k, len(self.corpus_idx_to_id)),
            )

            use_prior = self.prior_weight > 0
            if use_prior:
                query_id_to_str = {v: k for k, v in query_tokenized.vocab.items()}

            for qi, qid in enumerate(query_ids):
                doc_indices = queries_results[qi]
                bm25_scores = queries_scores[qi].astype(np.float64)

                query_documents = (
                    top_ranked[qid] if top_ranked and qid in top_ranked else None
                )

                doc_id_to_score: dict[str, float] = {}

                # Separate positive-score candidates for Bayesian re-scoring
                positive_mask = bm25_scores > 0
                positive_indices = np.where(positive_mask)[0]

                if len(positive_indices) > 0:
                    cand_doc_indices = doc_indices[positive_indices]
                    cand_bm25_scores = bm25_scores[positive_indices]

                    # Dynamic beta = median of BM25 scores for this query
                    beta = float(np.median(cand_bm25_scores))

                    # Dynamic alpha scaling for query-level score distribution
                    # invariance. The paper defines alpha as sigmoid steepness;
                    # dividing by std(scores) keeps the effective steepness
                    # consistent across queries whose BM25 ranges vary widely,
                    # preventing sigmoid saturation on high-scoring queries.
                    score_std = float(np.std(cand_bm25_scores))
                    alpha_eff = (
                        self.alpha / score_std if score_std > 1e-10 else self.alpha
                    )

                    # Sigmoid likelihood (monotonic -- preserves BM25 ranking)
                    x = np.clip(alpha_eff * (cand_bm25_scores - beta), -500, 500)
                    likelihood = 1.0 / (1.0 + np.exp(-x))

                    if use_prior:
                        # Map query token IDs to corpus vocab IDs
                        query_term_strs = [
                            query_id_to_str[tid] for tid in query_tokenized.ids[qi]
                        ]
                        query_corpus_ids = np.array(
                            [
                                self.corpus_vocab[t]
                                for t in query_term_strs
                                if t in self.corpus_vocab
                            ],
                            dtype=np.int32,
                        )

                        n_cand = len(cand_doc_indices)
                        query_tfs = np.zeros(n_cand, dtype=np.float64)
                        cand_doc_lengths = np.zeros(n_cand, dtype=np.float64)

                        has_query_terms = len(query_corpus_ids) > 0
                        for ci in range(n_cand):
                            doc_idx = cand_doc_indices[ci]
                            cand_doc_lengths[ci] = self.doc_lengths[doc_idx]
                            if has_query_terms:
                                doc_ids = self.corpus_token_ids[doc_idx]
                                query_tfs[ci] = np.isin(doc_ids, query_corpus_ids).sum()

                        composite = _composite_prior(
                            query_tfs, cand_doc_lengths, self.avg_dl
                        )
                        # Interpolate between flat prior (0.5) and composite
                        prior = 0.5 + self.prior_weight * (composite - 0.5)
                        posterior = (likelihood * prior) / (
                            likelihood * prior + (1.0 - likelihood) * (1.0 - prior)
                        )
                    else:
                        # prior=0.5 => posterior = likelihood
                        posterior = likelihood

                    for ci in range(len(cand_doc_indices)):
                        doc_id = self.corpus_idx_to_id[cand_doc_indices[ci]]
                        if query_documents is None or doc_id in query_documents:
                            doc_id_to_score[doc_id] = float(posterior[ci])

                # Include zero-score documents with score 0.0
                for vi in np.where(~positive_mask)[0]:
                    doc_id = self.corpus_idx_to_id[doc_indices[vi]]
                    if query_documents is None or doc_id in query_documents:
                        doc_id_to_score[doc_id] = 0.0

                results[qid] = doc_id_to_score

            return results

    return BB25Search(**kwargs)


bb25_model = ModelMeta(
    loader=bb25_loader,
    name="baseline/bb25",
    model_type=["sparse"],
    languages=None,
    open_weights=True,
    revision="0_1_1",
    release_date="2026-02-06",
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    embed_dim=None,
    license=None,
    max_tokens=None,
    reference="https://github.com/instructkr/bb25",
    similarity_fn_name=None,
    framework=[],
    use_instructions=False,
    public_training_code="https://github.com/instructkr/bb25",
    public_training_data=None,
    training_datasets=None,
    citation="""@software{jeong2026bayesianbm25,
  title={Bayesian BM25: A Probabilistic Framework for Hybrid Text and Vector Search},
  author={Jeong, Jaepil},
  year={2026},
  doi={10.5281/zenodo.18414941},
  url={https://doi.org/10.5281/zenodo.18414941},
}
@software{jeong2026neural,
  title={From Bayesian Inference to Neural Computation: The Analytical Emergence of Neural Network Structure from Probabilistic Relevance Estimation},
  author={Jeong, Jaepil},
  year={2026},
  doi={10.5281/zenodo.18512411},
  url={https://doi.org/10.5281/zenodo.18512411},
}""",
)

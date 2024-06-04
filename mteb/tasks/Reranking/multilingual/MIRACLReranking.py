from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import tqdm
from datasets import Dataset

from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from mteb.evaluation.evaluators import RerankingEvaluator
from mteb.evaluation.evaluators.RetrievalEvaluator import RetrievalEvaluator
from mteb.evaluation.evaluators.utils import cos_sim
from mteb.MTEBResults import ScoresDict

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskReranking import AbsTaskReranking

logger = logging.getLogger(__name__)

_EVAL_SPLIT = "dev"
_LANGUAGES = {
    "ar": ["ara-Arab"],
    "bn": ["ben-Beng"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fa": ["fas-Arab"],
    "fi": ["fin-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "id": ["ind-Latn"],
    "ja": ["jpn-Jpan"],
    "ko": ["kor-Kore"],
    "ru": ["rus-Cyrl"],
    "sw": ["swa-Latn"],
    "te": ["tel-Telu"],
    "th": ["tha-Thai"],
    "yo": ["yor-Latn"],
    "zh": ["zho-Hans"],
}

_CITATION = """@article{10.1162/tacl_a_00595,
    author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
    title = "{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {11},
    pages = {1114-1131},
    year = {2023},
    month = {09},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00595},
}"""


class MIRACLReranking(MultilingualTask, AbsTaskReranking):
    metadata = TaskMetadata(
        name="MIRACLReranking",
        description="MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages.",
        reference="https://project-miracl.github.io/",
        dataset={
            "path": "miracl/mmteb-miracl-reranking",
            "revision": "6d1962c527217f8927fca80f890f14f36b2802af",
        },
        type="Reranking",
        category="s2s",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGUAGES,
        main_score="NDCG@10(MIRACL)",
        date=("2022-06-01", "2023-01-30"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=[],
        license="CC BY-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation=_CITATION,
        n_samples={"dev": 44608},
        avg_character_length={"dev": 506.30},
    )

    def _evaluate_subset(
        self,
        model: Encoder | EncoderWithQueryCorpusEncode,
        data_split: Dataset,
        **kwargs: Any,
    ) -> ScoresDict:
        evaluator = MIRACLRerankingEvaluator(data_split, **kwargs)
        scores = evaluator(model)

        self._add_main_score(scores)
        return scores


class MIRACLRerankingEvaluator(RerankingEvaluator):
    """This class evaluates a SentenceTransformer model for the task of re-ranking.
    MIRACLRerankingEvaluator differs from RerankingEvaluator in two ways:
    1. it uses the pytrec_eval via RetrievalEvaluator instead of the metrics provided by sklearn;
    2. it reranks the top-k `candidates` from previous-stage retrieval which may not include all ground-truth `positive` documents
    """

    def __init__(
        self,
        samples: list[dict],
        mrr_at_k: int = 10,
        name: str = "",
        similarity_fct=cos_sim,
        batch_size: int = 512,
        use_batched_encoding: bool = True,
        limit: int | None = None,
        k_values: list[int] = [1, 3, 5, 10, 20, 100, 1000],
        **kwargs,
    ):
        """Args:
        k_values: ranking cutoff threshold when applicable
        """
        super().__init__(
            samples,
            mrr_at_k,
            name,
            similarity_fct,
            batch_size,
            use_batched_encoding,
            limit,
            **kwargs,
        )
        self.k_values = k_values

    def rerank(
        self, query_emb: torch.Tensor, docs_emb: torch.Tensor
    ) -> dict[str, float]:
        """Rerank documents (docs_emb) given the query (query_emb)

        Args:
            query_emb: Query embedding of shape `(num_queries, hidden_size)`)
                if `num_queries` > 0: we take the closest document to any of the queries
            docs_emb: Candidates documents embeddings of shape `(num_pos+num_neg, hidden_size)`)

        Returns:
            similarity_scores:
        """
        if not query_emb.shape[0]:
            raise ValueError("Empty query embedding")

        if not docs_emb.shape[0]:
            return {"empty-docid": 0}

        pred_scores = self.similarity_fct(query_emb, docs_emb)
        if len(pred_scores.shape) > 1:
            pred_scores = torch.amax(pred_scores, dim=0)

        return {
            str(i): score.detach().numpy().item() for i, score in enumerate(pred_scores)
        }

    def compute_metrics_batched(self, model: Encoder | EncoderWithQueryCorpusEncode):
        """Computes the metrices in a batched way, by batching all queries and
        all documents together
        """
        # using encode_queries and encode_corpus functions if they exists,
        # which can be defined by users to add different instructions for query and passage conveniently
        encode_queries_func = (
            model.encode_queries
            if isinstance(model, EncoderWithQueryCorpusEncode)
            else model.encode
        )
        encode_corpus_func = (
            model.encode_corpus
            if isinstance(model, EncoderWithQueryCorpusEncode)
            else model.encode
        )

        logger.info("Encoding queries...")
        if isinstance(self.samples[0]["query"], str):
            all_query_embs = np.asarray(
                encode_queries_func(
                    [sample["query"] for sample in self.samples],
                    batch_size=self.batch_size,
                )
            )
        elif isinstance(self.samples[0]["query"], list):
            # In case the query is a list of strings, we get the most similar embedding to any of the queries
            all_query_flattened = [
                q for sample in self.samples for q in sample["query"]
            ]
            all_query_embs = np.asarray(
                encode_queries_func(all_query_flattened, batch_size=self.batch_size)
            )
        else:
            raise ValueError(
                f"Query must be a string or a list of strings but is {type(self.samples[0]['query'])}"
            )

        logger.info("Encoding candidates...")
        all_docs = []
        for sample in self.samples:
            all_docs.extend(sample["candidates"])

        all_docs_embs = np.asarray(
            encode_corpus_func(all_docs, batch_size=self.batch_size)
        )

        # Compute scores
        logger.info("Evaluating...")
        query_idx, docs_idx = 0, 0
        results, qrels = {}, {}
        for instance in self.samples:
            num_subqueries = (
                len(instance["query"]) if isinstance(instance["query"], list) else 1
            )
            query_emb = all_query_embs[query_idx : query_idx + num_subqueries]
            query_idx += num_subqueries

            positive = instance["positive"]
            docs = instance["candidates"]
            num_doc = len(docs)
            docs_emb = all_docs_embs[docs_idx : docs_idx + num_doc]
            docs_idx += num_doc

            fake_qid = str(query_idx)
            results[fake_qid] = self.rerank(query_emb, docs_emb)
            qrels[fake_qid] = {
                str(i): 1 if doc in positive else 0 for i, doc in enumerate(docs)
            }

        ndcg, _map, recall, precision, naucs = RetrievalEvaluator.evaluate(
            qrels=qrels,
            results=results,
            k_values=self.k_values,
            ignore_identical_ids=False,
        )
        scores = {**ndcg, **_map, **recall, **precision, **naucs}
        scores_miracl = {f"{k}(MIRACL)": v for k, v in scores.items()}
        return scores_miracl

    def compute_metrics_individual(self, model):
        """Embeds every (query, positive, negative) tuple individually.
        Is slower than the batched version, but saves memory as only the
        embeddings for one tuple are needed. Useful when you have
        a really large test set
        """
        # using encode_queries and encode_corpus functions if they exists,
        # which can be defined by users to add different instructions for query and passage conveniently
        encode_queries_func = (
            model.encode_queries if hasattr(model, "encode_queries") else model.encode
        )
        encode_corpus_func = (
            model.encode_corpus if hasattr(model, "encode_corpus") else model.encode
        )

        results, qrels = {}, {}
        for i, instance in enumerate(tqdm.tqdm(self.samples, desc="Samples")):
            query = instance["query"]
            positive = set(instance["positive"])
            docs = list(instance["candidates"])

            if isinstance(query, str):
                # .encoding interface requires List[str] as input
                query_emb = np.asarray(
                    encode_queries_func([query], batch_size=self.batch_size)
                )
                docs_emb = np.asarray(
                    encode_corpus_func(docs, batch_size=self.batch_size)
                )

            fake_qid = str(i)
            results[fake_qid] = self.rerank(query_emb, docs_emb)
            qrels[fake_qid] = {
                str(i): 1 if doc in positive else 0 for i, doc in enumerate(docs)
            }

        ndcg, _map, recall, precision, naucs = RetrievalEvaluator.evaluate(
            qrels=qrels,
            results=results,
            k_values=self.k_values,
            ignore_identical_ids=False,
        )
        scores = {**ndcg, **_map, **recall, **precision, **naucs}
        scores_miracl = {f"{k}(MIRACL)": v for k, v in scores.items()}
        return scores_miracl

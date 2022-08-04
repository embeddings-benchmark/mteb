import logging
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm, trange

from .Evaluator import Evaluator
from .utils import cos_sim, dot_score


logger = logging.getLogger(__name__)


class RetrievalEvaluator(Evaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.
    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)
    """

    def __init__(
        self,
        queries: Dict[str, str],  # qid => query
        corpus: Dict[str, str],  # cid => doc
        relevant_docs: Dict[str, Set[str]],  # qid => Set[cid]
        corpus_chunk_size: int = 50000,
        mrr_at_k: List[int] = [10],
        ndcg_at_k: List[int] = [10],
        accuracy_at_k: List[int] = [1, 3, 5, 10],
        precision_recall_at_k: List[int] = [1, 3, 5, 10],
        map_at_k: List[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        score_functions: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
            "cos_sim": cos_sim,
            "dot": dot_score,
        },  # Score function, higher=more similar
        main_score_function: str = None,
        limit: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)
                if limit and len(self.queries_ids) >= limit:
                    break

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = main_score_function

    def __call__(self, model) -> float:
        scores = self.compute_metrics(model)

        if self.main_score_function is None:
            scores["main_score"] = max(
                [scores[name]["map_at_" + str(max(self.map_at_k))] for name in self.score_function_names]
            )
        else:
            scores["main_score"] = scores[self.main_score_function]["map_at_" + str(max(self.map_at_k))]
        return scores

    def compute_metrics(self, model, corpus_model=None, corpus_embeddings: torch.Tensor = None) -> Dict[str, float]:
        if corpus_model is None:
            corpus_model = model

        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.accuracy_at_k),
            max(self.precision_recall_at_k),
            max(self.map_at_k),
        )

        # Compute embedding for the queries
        logger.info("Encoding the queries...")
        query_embeddings = model.encode(
            self.queries,
            show_progress_bar=self.show_progress_bar,
            batch_size=self.batch_size,
            convert_to_tensor=True,
        )

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        # Iterate over chunks of the corpus
        logger.info("Encoding chunks of corpus, and computing similarity scores with queries...")
        for corpus_start_idx in trange(
            0,
            len(self.corpus),
            self.corpus_chunk_size,
            desc="Corpus Chunks",
            disable=not self.show_progress_bar,
        ):

            # Encode chunk of corpus
            if corpus_embeddings is None:
                corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))
                sub_corpus_embeddings = corpus_model.encode(
                    self.corpus[corpus_start_idx:corpus_end_idx],
                    show_progress_bar=False,
                    batch_size=self.batch_size,
                    convert_to_tensor=True,
                )
            else:
                corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus_embeddings))
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            # Compute cosine similarites
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                # Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                    pair_scores,
                    min(max_k, len(pair_scores[0])),
                    dim=1,
                    largest=True,
                    sorted=False,
                )
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(
                        pair_scores_top_k_idx[query_itr],
                        pair_scores_top_k_values[query_itr],
                    ):
                        corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
                        queries_result_list[name][query_itr].append({"corpus_id": corpus_id, "score": score})

        # Compute scores
        logger.info("Computing metrics...")
        scores = {name: self._compute_metrics(queries_result_list[name]) for name in self.score_functions}

        return scores

    def _compute_metrics(self, queries_result_list: List[object]):
        """
        Compute metrics for a list of queries

        Args:
            queries_result_list (`List[List[Dict]]`): List of lists of dictionaries with keys "corpus_id" and "score"

        Returns:
            `Dict[str, Dict[str, float]]`: Dictionary with keys "mrr@k", "ndcg@k", "accuracy@k", "precision_recall@k", "map@k"
                which values are dictionaries with scores for different k values
        """
        # Init score computation values
        num_hits_at_k = {"accuracy_at_" + str(k): 0 for k in self.accuracy_at_k}
        precisions_at_k = {"precision_at_" + str(k): [] for k in self.precision_recall_at_k}
        recall_at_k = {"recall_at_" + str(k): [] for k in self.precision_recall_at_k}
        MRR = {"mrr_at_" + str(k): 0 for k in self.mrr_at_k}
        ndcg = {"ndcg_at_" + str(k): [] for k in self.ndcg_at_k}
        AveP_at_k = {"map_at_" + str(k): [] for k in self.map_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x["score"], reverse=True)
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is accross the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_hits_at_k["accuracy_at_" + str(k_val)] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k["precision_at_" + str(k_val)].append(num_correct / k_val)
                recall_at_k["recall_at_" + str(k_val)].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        MRR["mrr_at_" + str(k_val)] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["corpus_id"] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(
                    true_relevances, k_val
                )
                ndcg["ndcg_at_" + str(k_val)].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k["map_at_" + str(k_val)].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        return {**num_hits_at_k, **precisions_at_k, **recall_at_k, **MRR, **ndcg, **AveP_at_k}

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg

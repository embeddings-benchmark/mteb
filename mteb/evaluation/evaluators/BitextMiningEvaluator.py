from __future__ import annotations

import logging

import torch
import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from mteb.encoder_interface import Encoder

from .Evaluator import Evaluator
from .normalize_encode import model_encode
from .utils import cos_sim

logger = logging.getLogger(__name__)


class BitextMiningEvaluator(Evaluator):
    def __init__(self, sentences, task_name: str, subsets=None, **kwargs):
        super().__init__(**kwargs)
        # By default, all the columns in sentences will serve for evaluation
        # Specifying a 'subsets' attribute will limit to certain columns
        self.subsets = (
            subsets if subsets is not None else list(sentences.features.keys())
        )
        self.n = len(sentences)
        self.n_subsets = len(self.subsets)
        self.sentences = sentences
        self.gold = (
            list(zip(range(self.n), range(self.n)))
            if "gold" not in sentences
            else sentences["gold"]
        )
        self.task_name = task_name

    def __call__(self, model: Encoder):
        scores = self.compute_metrics(model)
        return scores

    def compute_metrics(self, model: Encoder):
        # Compute embeddings
        logger.info(f"Encoding {self.n_subsets}x{self.n} sentences")
        embeddings = {}
        for sub in tqdm.tqdm(
            self.subsets, desc=f"Encoding {self.n_subsets}x{self.n} sentences"
        ):
            embeddings[sub] = model_encode(
                self.sentences[sub], model=model, task_name=self.task_name
            )

        if set(self.subsets) == {"sentence1", "sentence2"}:  # Case of a single pair
            return self._compute_metrics(
                embeddings["sentence1"], embeddings["sentence2"]
            )

        # Otherwise evaluate all pairs
        scores = {}
        for i in tqdm.tqdm(range(self.n_subsets), desc="Matching sentences"):
            for j in range(i + 1, self.n_subsets):
                key1, key2 = self.subsets[i], self.subsets[j]
                embeddings1 = embeddings[key1]
                embeddings2 = embeddings[key2]
                scores[f"{key1}-{key2}"] = self._compute_metrics(
                    embeddings1, embeddings2
                )
        return scores

    def _compute_metrics(
        self,
        embeddings1,
        embeddings2,
    ):
        # Find nearest neighbors
        logger.info("Finding nearest neighbors...")
        nearest_neighbors = self._similarity_search(embeddings1, embeddings2, top_k=1)

        # Compute errors
        logger.info("Computing metrics...")
        labels = []
        predictions = []
        for i, x in enumerate(nearest_neighbors):
            j = x[0]["corpus_id"]
            predictions.append(j)
            labels.append(self.gold[i][1])

        scores = {
            "precision": precision_score(
                labels, predictions, zero_division=0, average="weighted"
            ),
            "recall": recall_score(
                labels, predictions, zero_division=0, average="weighted"
            ),
            "f1": f1_score(labels, predictions, zero_division=0, average="weighted"),
            "accuracy": accuracy_score(labels, predictions),
        }
        return scores

    def _similarity_search(
        self,
        query_embeddings,
        corpus_embeddings,
        query_chunk_size: int = 100,
        corpus_chunk_size: int = 500000,
        top_k: int = 10,
        score_function=cos_sim,
    ):
        """This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
        It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

        Args:
            query_embeddings: A 2 dimensional tensor with the query embeddings.
            corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
            query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
            corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
            top_k: Retrieve top k matching entries.
            score_function: Function for computing scores. By default, cosine similarity.

        Returns:
            Returns a list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
        """
        query_embeddings = torch.from_numpy(query_embeddings)
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.unsqueeze(0)
        if len(corpus_embeddings.shape) == 1:
            corpus_embeddings = corpus_embeddings.unsqueeze(0)

        # Check that corpus and queries are on the same device
        if corpus_embeddings.device != query_embeddings.device:
            query_embeddings = query_embeddings.to(corpus_embeddings.device)

        queries_result_list = [[] for _ in range(len(query_embeddings))]

        for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
            # Iterate over chunks of the corpus
            for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
                # Compute cosine similarities
                cos_scores = score_function(
                    query_embeddings[
                        query_start_idx : query_start_idx + query_chunk_size
                    ],
                    corpus_embeddings[
                        corpus_start_idx : corpus_start_idx + corpus_chunk_size
                    ],
                )

                # Get top-k scores
                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                    cos_scores,
                    min(top_k, len(cos_scores[0])),
                    dim=1,
                    largest=True,
                    sorted=False,
                )
                cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
                cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(cos_scores)):
                    for sub_corpus_id, score in zip(
                        cos_scores_top_k_idx[query_itr],
                        cos_scores_top_k_values[query_itr],
                    ):
                        corpus_id = corpus_start_idx + sub_corpus_id
                        query_id = query_start_idx + query_itr
                        queries_result_list[query_id].append(
                            {"corpus_id": corpus_id, "score": score}
                        )

        # Sort and strip to top_k results
        for idx in range(len(queries_result_list)):
            queries_result_list[idx] = sorted(
                queries_result_list[idx], key=lambda x: x["score"], reverse=True
            )
            queries_result_list[idx] = queries_result_list[idx][0:top_k]

        return queries_result_list

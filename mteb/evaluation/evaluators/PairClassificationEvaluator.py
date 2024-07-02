from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

from mteb.encoder_interface import Encoder, EncoderWithSimilarity
from mteb.evaluation.evaluators.model_encode import model_encode

from .Evaluator import Evaluator

logger = logging.getLogger(__name__)


class PairClassificationEvaluator(Evaluator):
    """Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.
    The results are written in a CSV. If a CSV already exists, then values are appended.
    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

    Args:
        sentences1: The first column of sentences
        sentences2: The second column of sentences
        labels: labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1
        name: Name for the output
        batch_size: Batch size used to compute embeddings
        write_csv: Write results to a CSV file
    """

    def __init__(
        self,
        sentences1,
        sentences2,
        labels,
        task_name: str | None = None,
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit:
            sentences1 = sentences1[:limit]
            sentences2 = sentences2[:limit]
            labels = labels[:limit]
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.task_name = task_name

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)
        for label in labels:
            assert label == 0 or label == 1

    def __call__(
        self,
        model: Encoder | EncoderWithSimilarity,
        encode_kwargs: dict[str, Any] = {},
    ):
        scores = self.compute_metrics(model, encode_kwargs=encode_kwargs)

        # Main score is the max of Average Precision (AP)
        main_score = max(scores[short_name]["ap"] for short_name in scores)
        scores["main_score"] = main_score
        return scores

    def compute_metrics(
        self,
        model: Encoder | EncoderWithSimilarity,
        *,
        encode_kwargs: dict[str, Any] = {},
    ):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 32

        sentences = list(set(self.sentences1 + self.sentences2))

        total_sents = len(self.sentences1) + len(self.sentences2)
        n_duplicates = total_sents - len(sentences)
        if n_duplicates:
            logger.warning(
                f"Found {n_duplicates}/{total_sents} duplicates in the input data. Only encoding unique sentences."
            )
        embeddings = model_encode(
            sentences,
            model=model,
            prompt_name=self.task_name,
            **encode_kwargs,
        )
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings1 = [emb_dict[sent] for sent in self.sentences1]
        embeddings2 = [emb_dict[sent] for sent in self.sentences2]

        logger.info("Computing similarity distances.")
        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

        if hasattr(model, "similarity_pairwise"):
            similarity_scores = model.similarity_pairwise(embeddings1, embeddings2)  # type: ignore
        elif hasattr(model, "similarity"):
            _similarity_scores = [
                float(model.similarity(e1, e2))  # type: ignore
                for e1, e2 in zip(embeddings1, embeddings2)
            ]
            similarity_scores = np.array(_similarity_scores)
        else:
            similarity_scores = cosine_scores  # Default to cosine similarity

        embeddings1_np = np.asarray(embeddings1)
        embeddings2_np = np.asarray(embeddings2)
        dot_scores = [
            np.dot(embeddings1_np[i], embeddings2_np[i])
            for i in range(len(embeddings1_np))
        ]

        logger.info("Computing metrics...")
        labels = np.asarray(self.labels)
        output_scores = {}
        for short_name, name, scores, reverse in [
            ["similarity", "Model-Specified Similarity", similarity_scores, True],
            ["cosine", "Cosine-Similarity", cosine_scores, True],
            ["manhattan", "Manhattan-Distance", manhattan_distances, False],
            ["euclidean", "Euclidean-Distance", euclidean_distances, False],
            ["dot", "Dot-Product", dot_scores, True],
        ]:
            output_scores[short_name] = self._compute_metrics(scores, labels, reverse)

        return output_scores

    @staticmethod
    def _compute_metrics(
        scores: np.ndarray, labels: np.ndarray, high_score_more_similar: bool
    ) -> dict[str, float]:
        """Compute the metrics for the given scores and labels.

        Args:
            scores: The similarity/dissimilarity scores for the pairs, specified as an array of shape (n_pairs, ).
            labels: The labels for the pairs, specified as an array of shape (n_pairs, ).
            high_score_more_similar: If true, then the higher the score, the more similar the pairs are.

        Returns:
            The metrics for the given scores and labels.
        """
        acc, acc_threshold = PairClassificationEvaluator.find_best_acc_and_threshold(
            scores, labels, high_score_more_similar
        )
        f1, precision, recall, f1_threshold = (
            PairClassificationEvaluator.find_best_f1_and_threshold(
                scores, labels, high_score_more_similar
            )
        )
        ap = PairClassificationEvaluator.ap_score(
            scores, labels, high_score_more_similar
        )

        return {
            "accuracy": float(acc),
            "accuracy_threshold": float(acc_threshold),
            "f1": float(f1),
            "f1_threshold": float(f1_threshold),
            "precision": float(precision),
            "recall": float(recall),
            "ap": float(ap),
        }

    @staticmethod
    def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = sum(np.array(labels) == 0)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return max_acc, best_threshold

    @staticmethod
    def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0
        nextract = 0
        ncorrect = 0
        total_num_duplicates = sum(labels)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            nextract += 1

            if label == 1:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return best_f1, best_precision, best_recall, threshold

    @staticmethod
    def ap_score(scores, labels, high_score_more_similar: bool):
        return average_precision_score(
            labels, scores * (1 if high_score_more_similar else -1)
        )

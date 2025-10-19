import hashlib
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import average_precision_score

from mteb._evaluators import PairClassificationEvaluator
from mteb._evaluators.pair_classification_evaluator import (
    PairClassificationDistances,
)
from mteb.abstasks._statistics_calculation import (
    calculate_image_statistics,
    calculate_label_statistics,
    calculate_text_statistics,
)
from mteb.abstasks.abstask import AbsTask
from mteb.models.model_meta import ScoringFunction
from mteb.models.models_protocols import EncoderProtocol
from mteb.types.statistics import (
    ImageStatistics,
    LabelStatistics,
    SplitDescriptiveStatistics,
    TextStatistics,
)

logger = logging.getLogger(__name__)


class PairClassificationDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for PairClassification

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.
        unique_text_pairs: Number of unique pairs

        text1_statistics: Statistics for sentence1
        text2_statistics: Statistics for sentence2
        labels_statistics: Statistics for labels
    """

    num_samples: int
    number_of_characters: int
    unique_pairs: int

    text1_statistics: TextStatistics | None
    image1_statistics: ImageStatistics | None
    text2_statistics: TextStatistics | None
    image2_statistics: ImageStatistics | None
    labels_statistics: LabelStatistics


class AbsTaskPairClassification(AbsTask):
    """Abstract class for PairClassificationTasks

    The similarity is computed between pairs and the results are ranked. Average precision
    is computed to measure how well the methods can be used for pairwise pair classification.

    Attributes:
        dataset: A HuggingFace dataset containing the data for the task. Should contain the following columns: sentence1, sentence2, labels.
        input1_column_name: The name of the column containing the first sentence in the pair.
        input2_column_name: The name of the column containing the second sentence in the pair.
        label_column_name: The name of the column containing the labels for the pairs. Labels should be 0 or 1.
        abstask_prompt: Prompt to use for the task for instruction model if not prompt is provided in TaskMetadata.prompt.
    """

    abstask_prompt = "Retrieve text that are semantically similar to the given text."
    input1_column_name: str = "sentence1"
    input2_column_name: str = "sentence2"
    label_column_name: str = "labels"

    def _evaluate_subset(
        self,
        model: EncoderProtocol,
        data_split: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, str],
        prediction_folder: Path | None = None,
        **kwargs,
    ) -> dict[str, float]:
        if self.metadata.modalities == ["text"]:
            # for compatibility with v1 version where datasets were stored in a single row
            data_split = data_split[0] if len(data_split) == 1 else data_split
        evaluator = PairClassificationEvaluator(
            data_split,
            self.input1_column_name,
            self.input2_column_name,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )
        similarity_scores = evaluator(model, encode_kwargs=encode_kwargs)

        if prediction_folder:
            self._save_task_predictions(
                similarity_scores,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )
        return self._compute_metrics(
            similarity_scores, data_split[self.label_column_name]
        )

    def _compute_metrics(
        self, similarity_scores: PairClassificationDistances, labels: list[int]
    ) -> dict[str, float]:
        logger.info("Computing metrics...")
        labels = np.asarray(labels)
        output_scores = {}
        max_scores = defaultdict(list)
        for short_name, scores, reverse in [
            [
                "similarity",
                similarity_scores["similarity_scores"],
                True,
            ],
            [ScoringFunction.COSINE.value, similarity_scores["cosine_scores"], True],
            [
                ScoringFunction.MANHATTAN.value,
                similarity_scores["manhattan_distances"],
                False,
            ],
            [
                ScoringFunction.EUCLIDEAN.value,
                similarity_scores["euclidean_distances"],
                False,
            ],
            [ScoringFunction.DOT_PRODUCT.value, similarity_scores["dot_scores"], True],
        ]:
            metrics = self._compute_metrics_values(scores, labels, reverse)
            for metric_name, metric_value in metrics.items():
                output_scores[f"{short_name}_{metric_name}"] = metric_value
                max_scores[metric_name].append(metric_value)

        for metric in max_scores:
            if metric in ["f1", "ap", "precision", "recall", "accuracy"]:
                output_scores[f"max_{metric}"] = max(max_scores[metric])
        return output_scores

    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> PairClassificationDescriptiveStatistics:
        if hf_subset:
            dataset = self.dataset[hf_subset][split]
        elif compute_overall:
            dataset = defaultdict(list)
            for hf_subset in self.metadata.eval_langs:
                cur_dataset = self.dataset[hf_subset][split]
                # for compatibility with v1 version where datasets were stored in a single row
                if isinstance(cur_dataset, list) or len(cur_dataset) == 1:
                    cur_dataset = cur_dataset[0]
                if isinstance(cur_dataset, Dataset):
                    for row in cur_dataset:
                        for k, v in row.items():
                            dataset[k].append(v)
                else:
                    for key, value in cur_dataset.items():
                        dataset[key].extend(value[0] if len(value) == 1 else value)
        else:
            dataset = self.dataset[split]

        if isinstance(dataset, list):
            dataset = dataset[0]

        input1 = (
            dataset[self.input1_column_name][0]
            if len(dataset[self.input1_column_name]) == 1
            else dataset[self.input1_column_name]
        )
        input2 = (
            dataset[self.input2_column_name][0]
            if len(dataset[self.input2_column_name]) == 1
            else dataset[self.input2_column_name]
        )
        labels = (
            dataset[self.label_column_name][0]
            if len(dataset[self.label_column_name]) == 1
            else dataset[self.label_column_name]
        )

        text1_statistics = None
        text2_statistics = None
        image1_statistics = None
        image2_statistics = None
        number_of_characters = None
        unique_pairs = None
        if self.metadata.modalities == ["text"]:
            text1_statistics = calculate_text_statistics(input1)
            text2_statistics = calculate_text_statistics(input2)
            number_of_characters = (
                text1_statistics["total_text_length"]
                + text2_statistics["total_text_length"]
            )
            unique_pairs = len(set(zip(input1, input2)))

        elif self.metadata.modalities == ["image"]:
            image1_statistics = calculate_image_statistics(input1)
            image2_statistics = calculate_image_statistics(input2)

            def _compute_image_hash(inputs: list) -> list[str]:
                hashes = set()
                for img in inputs:
                    img_bytes = img.tobytes()
                    img_hash = hashlib.md5(img_bytes).hexdigest()
                    hashes.add(img_hash)
                return list(hashes)

            image_1_hashes = _compute_image_hash(input1)
            image_2_hashes = _compute_image_hash(input2)
            unique_pairs = len(set(zip(image_1_hashes, image_2_hashes)))

        return PairClassificationDescriptiveStatistics(
            num_samples=len(input1),
            unique_pairs=unique_pairs,
            number_of_characters=number_of_characters,
            text1_statistics=text1_statistics,
            image1_statistics=image1_statistics,
            text2_statistics=text2_statistics,
            image2_statistics=image2_statistics,
            labels_statistics=calculate_label_statistics(labels),
        )

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        # previously pair classification datasets were stored in a single row
        if self.metadata.is_multilingual:
            for subset in self.dataset:
                for split in self.dataset[subset]:
                    if len(self.dataset[subset][split]) == 1:
                        self.dataset[subset][split] = self.dataset[subset][split][0]
        else:
            for split in self.dataset:
                if len(self.dataset[split]) == 1:
                    self.dataset[split] = self.dataset[split][0]
        self._upload_dataset_to_hub(
            repo_name,
            [
                self.input1_column_name,
                self.input2_column_name,
                self.label_column_name,
            ],
        )

    def _compute_metrics_values(
        self, scores: list[float], labels: np.ndarray, high_score_more_similar: bool
    ) -> dict[str, float]:
        """Compute the metrics for the given scores and labels.

        Args:
            scores: The similarity/dissimilarity scores for the pairs, specified as an array of shape (n_pairs, ).
            labels: The labels for the pairs, specified as an array of shape (n_pairs, ).
            high_score_more_similar: If true, then the higher the score, the more similar the pairs are.

        Returns:
            The metrics for the given scores and labels.
        """
        acc, acc_threshold = self._find_best_acc_and_threshold(
            scores, labels, high_score_more_similar
        )
        (
            f1,
            precision,
            recall,
            f1_threshold,
        ) = self._find_best_f1_and_threshold(scores, labels, high_score_more_similar)
        ap = average_precision_score(
            labels, np.array(scores) * (1 if high_score_more_similar else -1)
        )

        return dict(
            accuracy=float(acc),
            f1=float(f1),
            precision=float(precision),
            recall=float(recall),
            ap=float(ap),
        )

    def _find_best_acc_and_threshold(
        self, scores: np.ndarray, labels: np.ndarray, high_score_more_similar: bool
    ) -> tuple[float, float]:
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

    def _find_best_f1_and_threshold(
        self, scores, labels, high_score_more_similar: bool
    ) -> tuple[float, float, float, float]:
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

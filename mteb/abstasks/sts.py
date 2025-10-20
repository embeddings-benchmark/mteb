import logging
from pathlib import Path
from typing import Any, TypedDict, cast

from datasets import Dataset
from scipy.stats import pearsonr, spearmanr

from mteb._evaluators import AnySTSEvaluator
from mteb._evaluators.any_sts_evaluator import STSEvaluatorScores
from mteb.models import EncoderProtocol
from mteb.types.statistics import (
    ImageStatistics,
    ScoreStatistics,
    SplitDescriptiveStatistics,
    TextStatistics,
)

from ._statistics_calculation import (
    calculate_image_statistics,
    calculate_score_statistics,
    calculate_text_statistics,
)
from .abstask import AbsTask

logger = logging.getLogger(__name__)


class AnySTSDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for STS

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.
        unique_pairs: Number of unique pairs

        text1_statistics: Statistics for sentence1
        text2_statistics: Statistics for sentence2

        image1_statistics: Statistics for image1
        image2_statistics: Statistics for image2

        label_statistics: Statistics for labels
    """

    num_samples: int
    number_of_characters: int | None
    unique_pairs: int | None

    text1_statistics: TextStatistics | None
    text2_statistics: TextStatistics | None

    image1_statistics: ImageStatistics | None
    image2_statistics: ImageStatistics | None

    label_statistics: ScoreStatistics


class STSMetrics(TypedDict):
    """Metrics for STS tasks

    Attributes:
        pearson: Pearson correlation coefficient using the model's similarity function or cosine similarity if not available
        spearman: Spearman correlation coefficient using the model's similarity function or cosine similarity if not available
        cosine_pearson: Pearson correlation coefficient using cosine similarity
        cosine_spearman: Spearman correlation coefficient using cosine similarity
        manhattan_pearson: Pearson correlation coefficient using Manhattan distance
        manhattan_spearman: Spearman correlation coefficient using Manhattan distance
        euclidean_pearson: Pearson correlation coefficient using Euclidean distance
        euclidean_spearman: Spearman correlation coefficient using Euclidean distance
    """

    pearson: float
    spearman: float
    cosine_pearson: float
    cosine_spearman: float
    manhattan_pearson: float
    manhattan_spearman: float
    euclidean_pearson: float
    euclidean_spearman: float


class AbsTaskSTS(AbsTask):
    """Abstract class for STS experiments.

    Attributes:
        dataset: Dataset or dict of Datasets for different subsets (e.g., languages). Dataset must contain columns specified in column_names and a 'score' column.
            Columns in column_names should contain the text or image data to be compared.
        column_names: Tuple containing the names of the two columns to compare.
        min_score: Minimum possible score in the dataset.
        max_score: Maximum possible score in the dataset.
        abstask_prompt: Prompt to use for the task for instruction model if not prompt is provided in TaskMetadata.prompt.
    """

    abstask_prompt = "Retrieve semantically similar text."
    column_names: tuple[str, str] = ("sentence1", "sentence2")
    min_score: int = 0
    max_score: int = 5

    def _evaluate_subset(
        self,
        model: EncoderProtocol,
        data_split: Dataset,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        **kwargs: Any,
    ) -> STSMetrics:
        normalized_scores = list(map(self._normalize, data_split["score"]))
        data_split = data_split.select_columns(list(self.column_names))

        evaluator = AnySTSEvaluator(
            data_split,
            self.column_names,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )
        scores = evaluator(model, encode_kwargs=encode_kwargs)

        if prediction_folder:
            self._save_task_predictions(
                scores,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        return self._calculate_scores(scores, normalized_scores)

    def _calculate_scores(
        self, scores: STSEvaluatorScores, normalized_scores: list[float]
    ) -> STSMetrics:
        def compute_corr(x: list[float], y: list[float]) -> tuple[float, float]:
            """Return (pearson, spearman) correlations between x and y."""
            return pearsonr(x, y)[0], spearmanr(x, y)[0]

        cosine_pearson, cosine_spearman = compute_corr(
            normalized_scores, scores["cosine_scores"]
        )
        manhattan_pearson, manhattan_spearman = compute_corr(
            normalized_scores, scores["manhattan_distances"]
        )
        euclidean_pearson, euclidean_spearman = compute_corr(
            normalized_scores, scores["euclidean_distances"]
        )

        if scores["similarity_scores"] is not None:
            pearson, spearman = compute_corr(
                normalized_scores, scores["similarity_scores"]
            )
        else:
            # if model does not have a similarity function, assume cosine similarity
            pearson, spearman = cosine_pearson, cosine_spearman

        return STSMetrics(
            # using the models own similarity score
            pearson=pearson,
            spearman=spearman,
            # generic similarity scores
            cosine_pearson=cosine_pearson,
            cosine_spearman=cosine_spearman,
            manhattan_pearson=manhattan_pearson,
            manhattan_spearman=manhattan_spearman,
            euclidean_pearson=euclidean_pearson,
            euclidean_spearman=euclidean_spearman,
        )

    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> AnySTSDescriptiveStatistics:
        first_column, second_column = self.column_names
        self.dataset = cast(dict[str, dict[str, Dataset]], self.dataset)

        if hf_subset:
            sentence1 = self.dataset[hf_subset][split][first_column]
            sentence2 = self.dataset[hf_subset][split][second_column]
            score = self.dataset[hf_subset][split]["score"]
        elif compute_overall:
            sentence1 = []
            sentence2 = []
            score = []
            for hf_subset in self.metadata.eval_langs:
                sentence1.extend(self.dataset[hf_subset][split][first_column])
                sentence2.extend(self.dataset[hf_subset][split][second_column])
                score.extend(self.dataset[hf_subset][split]["score"])
        else:
            sentence1 = self.dataset[split][first_column]
            sentence2 = self.dataset[split][second_column]
            score = self.dataset[split]["score"]

        if "text" in self.metadata.modalities:
            text1_statistics = calculate_text_statistics(sentence1)
            text2_statistics = calculate_text_statistics(sentence2)

            unique_pairs = len(set(zip(sentence1, sentence2)))
        else:
            text1_statistics = None
            text2_statistics = None
            unique_pairs = None

        if "image" in self.metadata.modalities:
            image1_statistics = calculate_image_statistics(sentence1)
            image2_statistics = calculate_image_statistics(sentence2)
        else:
            image1_statistics = None
            image2_statistics = None

        labels_statistics = calculate_score_statistics(score)

        return AnySTSDescriptiveStatistics(
            num_samples=len(sentence1),
            number_of_characters=(
                text1_statistics["total_text_length"]
                + text2_statistics["total_text_length"]
                if text1_statistics
                else None
            ),
            unique_pairs=unique_pairs,
            text1_statistics=text1_statistics,
            text2_statistics=text2_statistics,
            image1_statistics=image1_statistics,
            image2_statistics=image2_statistics,
            label_statistics=labels_statistics,
        )

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        self._upload_dataset_to_hub(
            repo_name, [self.column_names[0], self.column_names[1], "score"]
        )

    def _normalize(self, x: float) -> float:
        return (x - self.min_score) / (self.max_score - self.min_score)

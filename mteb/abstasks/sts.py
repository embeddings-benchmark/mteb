from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypedDict, cast

from scipy.stats import pearsonr, spearmanr

from mteb._evaluators import AnySTSEvaluator
from mteb.models import EncoderProtocol
from mteb.types.statistics import (
    SplitDescriptiveStatistics,
)

from ._statistics_calculation import (
    calculate_pair_modality_statistics,
    calculate_score_statistics,
)
from .abstask import AbsTask

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from datasets import Dataset

    from mteb._evaluators.any_sts_evaluator import STSEvaluatorScores
    from mteb.models import MTEBModels
    from mteb.types import EncodeKwargs, Modalities, PromptType
    from mteb.types.statistics import (
        AudioStatistics,
        ImageStatistics,
        ScoreStatistics,
        TextStatistics,
        VideoStatistics,
    )

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

        audio1_statistics: Statistics for audio1
        audio2_statistics: Statistics for audio2

        video1_statistics: Statistics for video1
        video2_statistics: Statistics for video2

        label_statistics: Statistics for labels
    """

    num_samples: int
    number_of_characters: int | None
    unique_pairs: int | None

    text1_statistics: TextStatistics | None
    text2_statistics: TextStatistics | None

    image1_statistics: ImageStatistics | None
    image2_statistics: ImageStatistics | None

    audio1_statistics: AudioStatistics | None
    audio2_statistics: AudioStatistics | None

    video1_statistics: VideoStatistics | None
    video2_statistics: VideoStatistics | None

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
    """The class which semantic textual similarity (STS) tasks inherit from.

    A semantic textual similarity (STS) task consists of a dataset with pairs of sentences and corresponding similarity scores.
    The task is to predict the similarity score for each pair of sentences.

    The task works by encoding the sentences using the provided model and then calculating similarity scores using both the model-defined similarity
    function (if available) and generic similarity functions, including cosine similarity, Manhattan distance, and Euclidean distance.
    The predicted similarity scores are then compared to the true similarity scores using Pearson and Spearman correlation coefficients.


    Attributes:
        dataset: Dataset or dict of Datasets for different subsets (e.g., languages). Dataset must contain columns specified in column_names and a 'score' column.
            Columns in column_names should contain the text or image data to be compared.
        column_names: Tuple containing the names of the two columns to compare.
        min_score: Minimum possible score in the dataset.
        max_score: Maximum possible score in the dataset.
        abstask_prompt: Prompt to use for the task for instruction model if not prompt is provided in TaskMetadata.prompt.
        input1_prompt_type: Type of prompt of first input. Used for asymmetric tasks.
        input2_prompt_type: Type of prompt of second input. Used for asymmetric tasks.
    """

    abstask_prompt = "Retrieve semantically similar text."
    column_names: (
        tuple[str, str]
        | tuple[
            Sequence[tuple[str, Modalities]],
            Sequence[tuple[str, Modalities]],
        ]
    ) = (
        "sentence1",
        "sentence2",
    )
    min_score: int = 0
    max_score: int = 5
    input1_prompt_type: PromptType | None = None
    input2_prompt_type: PromptType | None = None

    def _evaluate_subset(
        self,
        model: MTEBModels,
        data_split: Dataset,
        *,
        encode_kwargs: EncodeKwargs,
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> STSMetrics:
        if not isinstance(model, EncoderProtocol):
            raise TypeError("Expected model to be an instance of EncoderProtocol")

        normalized_scores = list(map(self._normalize, data_split["score"]))

        evaluator = AnySTSEvaluator(
            data_split,
            self.column_names,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            input1_prompt_type=self.input1_prompt_type,
            input2_prompt_type=self.input2_prompt_type,
            **kwargs,
        )
        scores = evaluator(
            model,
            encode_kwargs=encode_kwargs,
            num_proc=num_proc,
        )

        if prediction_folder:
            self._save_task_predictions(
                scores,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        return self._calculate_scores(scores, normalized_scores)

    def _calculate_scores(  # noqa: PLR6301
        self, scores: STSEvaluatorScores, normalized_scores: list[float]
    ) -> STSMetrics:
        def compute_corr(x: list[float], y: list[float]) -> tuple[float, float]:
            """Return (pearson, spearman) correlations between x and y."""
            return float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0])

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
        self.dataset = cast("dict[str, dict[str, Dataset]]", self.dataset)

        # Pick a representative split dataset to inspect available column names.
        _ref_split: Dataset = (
            self.dataset[hf_subset][split]
            if hf_subset
            else (
                self.dataset[next(iter(self.metadata.eval_langs))][split]
                if compute_overall
                else self.dataset[split]
            )
        )
        if hf_subset:
            score = self.dataset[hf_subset][split]["score"]
            n = len(score)

            def _load_col(col: str) -> list:
                return self.dataset[hf_subset][split][col]  # type: ignore[index]

        elif compute_overall:
            score = []
            for _subset in self.metadata.eval_langs:
                score.extend(self.dataset[_subset][split]["score"])
            n = len(score)

            def _load_col(col: str) -> list:  # type: ignore[misc]
                result = []
                for subset in self.metadata.eval_langs:
                    result.extend(self.dataset[subset][split][col])
                return result

        else:
            score = self.dataset[split]["score"]
            n = len(score)

            def _load_col(col: str) -> list:  # type: ignore[misc]
                return self.dataset[split][col]

        if isinstance(self.column_names[0], str):
            modality1 = self.metadata.get_modalities(self.input1_prompt_type)[0]
            modality2 = self.metadata.get_modalities(self.input2_prompt_type)[0]
            col_modalities1: list[tuple[str, str]] = [(self.column_names[0], modality1)]  # type: ignore[index]
            col_modalities2: list[tuple[str, str]] = [(self.column_names[1], modality2)]  # type: ignore[index]
        else:
            col_modalities1 = list(self.column_names[0])  # type: ignore[arg-type]
            col_modalities2 = list(self.column_names[1])  # type: ignore[arg-type]

        pair_stats = calculate_pair_modality_statistics(
            col_modalities1,
            col_modalities2,
            _load_col,
            n,
        )
        labels_statistics = calculate_score_statistics(score)

        return AnySTSDescriptiveStatistics(
            num_samples=n,
            number_of_characters=(
                pair_stats["text1_statistics"]["total_text_length"]
                + pair_stats["text2_statistics"]["total_text_length"]
                if pair_stats["text1_statistics"]
                else None
            ),
            unique_pairs=pair_stats["unique_pairs"],
            text1_statistics=pair_stats["text1_statistics"],
            text2_statistics=pair_stats["text2_statistics"],
            image1_statistics=pair_stats["image1_statistics"],
            image2_statistics=pair_stats["image2_statistics"],
            audio1_statistics=pair_stats["audio1_statistics"],
            audio2_statistics=pair_stats["audio2_statistics"],
            video1_statistics=pair_stats["video1_statistics"],
            video2_statistics=pair_stats["video2_statistics"],
            label_statistics=labels_statistics,
        )

    def _push_dataset_to_hub(
        self,
        repo_name: str,
        num_proc: int | None = None,
    ) -> None:
        if isinstance(self.column_names[0], str):
            cols = [self.column_names[0], self.column_names[1]]
        else:
            cols = [col for col, _ in self.column_names[0]] + [
                col for col, _ in self.column_names[1]
            ]
        self._upload_dataset_to_hub(repo_name, [*cols, "score"], num_proc=num_proc)

    def _normalize(self, x: float) -> float:
        return (x - self.min_score) / (self.max_score - self.min_score)

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
from datasets import Dataset
from scipy.integrate import trapezoid
from sklearn.metrics.pairwise import euclidean_distances

from mteb._create_dataloaders import create_dataloader
from mteb.models import EncoderProtocol
from mteb.types.statistics import SpatialArrangementDescriptiveStatistics

from ._statistics_calculation import calculate_text_statistics
from .abstask import AbsTask

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from mteb.models import MTEBModels
    from mteb.timing import TimingStack
    from mteb.types import EncodeKwargs

logger = logging.getLogger(__name__)


class SpatialArrangementMetrics(TypedDict):
    """Metrics for SpatialArrangement tasks

    Attributes:
        alpha_auc: Area under the triplet-based Krippendorff's alpha curve, integrated over the log of the distance-ratio
            threshold and normalized to [-1, 1].
    """

    alpha_auc: float


def _triplet_indices(n: int) -> tuple[NDArray[np.int64], ...]:
    """All (anchor, i, j) triplets with i < j and anchor not in {i, j}."""
    i_idx, j_idx = np.triu_indices(n, k=1)
    anchors = np.arange(n)[None, :].repeat(len(i_idx), axis=0)
    valid = (anchors != i_idx[:, None]) & (anchors != j_idx[:, None])
    pair_idx, anchor = np.nonzero(valid)
    return anchor, i_idx[pair_idx], j_idx[pair_idx]


def _arrangement_triplets(
    coordinates: NDArray[np.floating], similarities: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Distance ratio and model agreement for every triplet in a single arrangement.

    Returns:
        ratio: distance from the anchor to the farther item divided by the distance to the closer item (human layout).
        agree: whether the model is strictly more similar to the item the human placed closer to the anchor.
    """
    distances = euclidean_distances(coordinates)
    anchor, i, j = _triplet_indices(len(coordinates))
    dist_i, dist_j = distances[anchor, i], distances[anchor, j]
    i_closer = dist_i < dist_j
    closer = np.where(i_closer, i, j)
    farther = np.where(i_closer, j, i)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.maximum(dist_i, dist_j) / np.minimum(dist_i, dist_j)
    agree = similarities[anchor, closer] > similarities[anchor, farther]
    return ratio, agree


def _alpha_auc(
    ratio: NDArray[np.floating],
    agree: NDArray[np.bool_],
    thresholds: NDArray[np.floating],
) -> float:
    """Normalized area under alpha(d) over log(d), where alpha(d) = 2 * agreement - 1 on triplets with ratio > d."""
    kept_thresholds, alphas = [], []
    for threshold in thresholds:
        mask = ratio > threshold
        if mask.any():
            kept_thresholds.append(threshold)
            alphas.append(2 * agree[mask].mean() - 1)
    log_thresholds = np.log(kept_thresholds)
    span = log_thresholds[-1] - log_thresholds[0]
    if span == 0:
        return float(np.mean(alphas))
    return float(trapezoid(alphas, log_thresholds) / span)


class AbsTaskSpatialArrangement(AbsTask):
    """Abstract class for spatial arrangement tasks (Spatial Arrangement Method, Goldstone 1994).

    Each sample is one arrangement: a rater placed a set of texts on a 2D canvas such that proximity reflects perceived
    similarity. Every (anchor, item, item) triplet in an arrangement defines which item the rater placed closer to
    the anchor. The model agrees with a triplet if it assigns the higher similarity to that item. Agreement is pooled
    over all arrangements and summarized as Krippendorff's alpha, alpha(d) = 2 * agreement - 1, on the triplets whose
    farther/closer distance ratio exceeds a threshold d. The main score is the area under alpha(d) over log(d).

    Attributes:
        dataset: Dataset or dict of Datasets for different subsets. Each row is one arrangement with the columns given
            by texts_column_name, x_column_name, y_column_name (aligned lists) and rater_column_name.
        texts_column_name: Name of the column with the list of arranged texts.
        x_column_name: Name of the column with the list of x coordinates.
        y_column_name: Name of the column with the list of y coordinates.
        rater_column_name: Name of the column identifying the rater.
        thresholds: Distance-ratio thresholds d over which alpha(d) is integrated.
        abstask_prompt: Prompt to use for the task for instruction model if not prompt is provided in TaskMetadata.prompt.
    """

    abstask_prompt = "Retrieve semantically similar text."
    texts_column_name: str = "texts"
    x_column_name: str = "x"
    y_column_name: str = "y"
    rater_column_name: str = "user_id"
    thresholds: NDArray[np.floating] = np.logspace(0, np.log10(6.5), 30)

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
        timer: TimingStack,
        **kwargs: Any,
    ) -> SpatialArrangementMetrics:
        if not isinstance(model, EncoderProtocol):
            raise TypeError("Expected model to be an instance of EncoderProtocol")

        arrangements = data_split[self.texts_column_name]
        unique_texts = list(dict.fromkeys(t for texts in arrangements for t in texts))
        text_to_idx = {text: idx for idx, text in enumerate(unique_texts)}

        with timer(
            "Encoding",
            split=hf_split,
            subset=hf_subset,
            log_message="Running spatial arrangement - Encoding samples...",
        ):
            embeddings = model.encode(
                create_dataloader(
                    Dataset.from_dict({"text": unique_texts}),
                    task_metadata=self.metadata,
                    input_column="text",
                    num_proc=num_proc,
                    **encode_kwargs,
                ),
                task_metadata=self.metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                **encode_kwargs,
            )

        with timer(
            "Scoring",
            split=hf_split,
            subset=hf_subset,
            log_message="Running spatial arrangement - Computing triplet agreement...",
        ):
            ratios, agreements = [], []
            for texts, x, y in zip(
                arrangements,
                data_split[self.x_column_name],
                data_split[self.y_column_name],
                strict=True,
            ):
                idx = [text_to_idx[text] for text in texts]
                similarities = np.asarray(
                    model.similarity(embeddings[idx], embeddings[idx]), dtype=float
                )
                ratio, agree = _arrangement_triplets(
                    np.column_stack([x, y]).astype(float), similarities
                )
                ratios.append(ratio)
                agreements.append(agree)

            alpha_auc = _alpha_auc(
                np.concatenate(ratios), np.concatenate(agreements), self.thresholds
            )
        return SpatialArrangementMetrics(alpha_auc=alpha_auc)

    def _calculate_descriptive_statistics_from_split(
        self,
        split: str,
        *,
        hf_subset: str | None = None,
        compute_overall: bool = False,
        num_proc: int | None = None,
    ) -> SpatialArrangementDescriptiveStatistics:
        self.dataset = cast("dict[str, dict[str, Dataset]]", self.dataset)

        if hf_subset:
            splits = [self.dataset[hf_subset][split]]
        elif compute_overall:
            splits = [
                self.dataset[subset][split] for subset in self.metadata.eval_langs
            ]
        else:
            splits = [self.dataset[split]]

        arrangements = [
            texts for data in splits for texts in data[self.texts_column_name]
        ]
        raters = {rater for data in splits for rater in data[self.rater_column_name]}
        all_texts = [text for texts in arrangements for text in texts]
        text_statistics = calculate_text_statistics(all_texts)

        return SpatialArrangementDescriptiveStatistics(
            num_samples=len(arrangements),
            number_of_characters=text_statistics["total_text_length"],
            num_raters=len(raters),
            num_triplets=sum(
                len(texts) * (len(texts) - 1) * (len(texts) - 2) // 2
                for texts in arrangements
            ),
            arranged_texts_statistics=text_statistics,
        )

    def _push_dataset_to_hub(
        self,
        repo_name: str,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._upload_dataset_to_hub(
            repo_name,
            [
                self.texts_column_name,
                self.x_column_name,
                self.y_column_name,
                self.rater_column_name,
            ],
            num_proc=num_proc,
            **kwargs,
        )

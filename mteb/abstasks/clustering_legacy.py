from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from datasets import Dataset
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

from mteb._evaluators import ClusteringEvaluator
from mteb.models import EncoderProtocol
from mteb.types.statistics import (
    SplitDescriptiveStatistics,
)

from ._statistics_calculation import (
    calculate_label_statistics,
    calculate_single_input_modality_statistics,
)
from .abstask import AbsTask

if TYPE_CHECKING:
    from pathlib import Path

    from mteb.models import MTEBModels
    from mteb.types import EncodeKwargs, ScoresDict
    from mteb.types.statistics import (
        AudioStatistics,
        ImageStatistics,
        LabelStatistics,
        TextStatistics,
        VideoStatistics,
    )

logger = logging.getLogger(__name__)


class ClusteringDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for Clustering

    Attributes:
        num_samples: number of samples in the dataset.

        text_statistics: Statistics for text
        image_statistics: Statistics for images
        audio_statistics: Statistics for audio
        video_statistics: Statistics for video
        label_statistics: Statistics for labels
    """

    num_samples: int

    text_statistics: TextStatistics | None
    image_statistics: ImageStatistics | None
    audio_statistics: AudioStatistics | None
    video_statistics: VideoStatistics | None
    label_statistics: LabelStatistics


class ClusteringMetrics(TypedDict, total=False):
    """Clustering metrics.

    Attributes:
        v_measure: V-measure score.
        nmi: Normalized Mutual Information score.
        ari: Adjusted Rand Index score.
        cluster_accuracy: Clustering accuracy score.
    """

    v_measure: float
    nmi: float
    ari: float
    cluster_accuracy: float


class AbsTaskClusteringLegacy(AbsTask):
    """Legacy abstract task for clustering. For new tasks, we recommend using AbsTaskClustering because it is faster, more sample-efficient, and produces more robust statistical estimates.

    Attributes:
        dataset: A HuggingFace Dataset containing the data for the clustering task. It must contain the following columns:
            sentences: List of inputs to be clustered. Can be text, images, etc. Name can be changed via `input_column_name`.
            labels: List of integer labels representing the true cluster assignments. Name can be changed via `label_column_name`.
        input_column_name: The name of the column in the dataset that contains the input sentences or data points.
        label_column_name: The name of the column in the dataset that contains the true cluster labels.
        abstask_prompt: Prompt to use for the task for instruction model if not prompt is provided in TaskMetadata.prompt.
    """

    abstask_prompt = "Identify categories in user passages."
    evaluator: type[ClusteringEvaluator] = ClusteringEvaluator
    input_column_name: str = "sentences"
    label_column_name: str = "labels"

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
    ) -> ScoresDict:
        if not isinstance(model, EncoderProtocol):
            raise TypeError("Expected model to be an instance of EncoderProtocol")

        data_split = data_split.select_columns(
            [self.input_column_name, self.label_column_name]
        )
        # MTEB text clustering requires renaming and eval per subset.
        if self.metadata.modalities == ["text"]:
            all_metrics = []
            clusters = []
            for i, cluster_set in enumerate(data_split):
                logger.info(
                    f"Running clustering on cluster ({i + 1}/{len(data_split)})"
                )
                labels = cluster_set[self.label_column_name]
                if len(set(labels)) <= 1:
                    logger.warning(
                        f"Cluster set {i} has {len(set(labels))} unique label(s), "
                        "skipping encode step — v_measure is always 1.0."
                    )
                    all_metrics.append(ClusteringMetrics(v_measure=1.0))
                    continue
                clustering_dataset = Dataset.from_dict(cluster_set).select_columns(
                    [self.input_column_name, self.label_column_name]
                )
                evaluator = self.evaluator(
                    clustering_dataset,
                    input_column_name=self.input_column_name,
                    label_column_name=self.label_column_name,
                    task_metadata=self.metadata,
                    hf_split=hf_split,
                    hf_subset=hf_subset,
                    **kwargs,
                )
                clusters_assignment = evaluator(model, encode_kwargs=encode_kwargs)
                clusters.append(clusters_assignment)
                set_metrics = self._compute_metrics(
                    clustering_dataset[self.label_column_name],
                    clusters_assignment,
                    v_measure_only=True,
                )
                all_metrics.append(set_metrics)

            if prediction_folder:
                self._save_task_predictions(
                    clusters,
                    model,
                    prediction_folder,
                    hf_subset=hf_subset,
                    hf_split=hf_split,
                )
            v_measures = [m["v_measure"] for m in all_metrics]
            v_mean = np.mean(v_measures)
            v_std = np.std(v_measures)
            scores = {
                "v_measure": v_mean,
                "v_measure_std": v_std,
                "v_measures": v_measures,
            }
            return scores

        evaluator = self.evaluator(
            data_split,
            input_column_name=self.input_column_name,
            label_column_name=self.label_column_name,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )
        evaluate_clusters = evaluator(
            model,
            encode_kwargs=encode_kwargs,
            num_proc=num_proc,
        )
        if prediction_folder:
            self._save_task_predictions(
                evaluate_clusters,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        return self._compute_metrics(
            data_split[self.label_column_name],
            evaluate_clusters,
        )

    def _compute_metrics(  # noqa: PLR6301
        self,
        labels: list[int],
        cluster_assignment: list[int],
        v_measure_only: bool = False,
    ) -> ClusteringMetrics:
        logger.info("Running clustering - Evaluating clustering...")
        v_measure = metrics.cluster.v_measure_score(labels, cluster_assignment)
        if v_measure_only:
            return ClusteringMetrics(
                v_measure=v_measure,
            )
        nmi = metrics.cluster.normalized_mutual_info_score(labels, cluster_assignment)
        ari = metrics.cluster.adjusted_rand_score(labels, cluster_assignment)

        matrix = metrics.confusion_matrix(labels, cluster_assignment)
        # get linear sum assignment
        row_ind, col_ind = linear_sum_assignment(matrix, maximize=True)
        total_correct = matrix[row_ind, col_ind].sum()
        clustering_accuracy = total_correct / len(labels)
        return ClusteringMetrics(
            v_measure=float(v_measure),
            nmi=float(nmi),
            ari=float(ari),
            cluster_accuracy=float(clustering_accuracy),
        )

    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> ClusteringDescriptiveStatistics:
        modality = self.metadata.modalities[0]
        if hf_subset:
            ds = self.dataset[hf_subset][split]
            col_inputs = {modality: ds[self.input_column_name]}
            labels = ds[self.label_column_name]
        elif compute_overall:
            col_inputs = {modality: []}
            labels = []
            for subset in self.metadata.eval_langs:
                ds = self.dataset[subset][split]
                col_inputs[modality].extend(ds[self.input_column_name])
                labels.extend(ds[self.label_column_name])
        else:
            ds = self.dataset[split]
            col_inputs = {modality: ds[self.input_column_name]}
            labels = ds[self.label_column_name]

        # Flatten nested lists (e.g. grouped sentence clusters)
        if col_inputs[modality] and isinstance(col_inputs[modality][0], list):
            col_inputs[modality] = [
                item for sublist in col_inputs[modality] for item in sublist
            ]
        if labels and isinstance(labels[0], list):
            labels = [item for sublist in labels for item in sublist]

        modality_stats = calculate_single_input_modality_statistics(col_inputs)
        return ClusteringDescriptiveStatistics(
            num_samples=len(col_inputs[modality]),
            **modality_stats,
            label_statistics=calculate_label_statistics(labels),
        )

    def _push_dataset_to_hub(
        self,
        repo_name: str,
        num_proc: int | None = None,
    ) -> None:
        self._upload_dataset_to_hub(
            repo_name,
            [
                self.input_column_name,
                self.label_column_name,
            ],
            num_proc=num_proc,
        )

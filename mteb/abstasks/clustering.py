import itertools
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import v_measure_score

from mteb._create_dataloaders import create_dataloader
from mteb.models import EncoderProtocol
from mteb.types import HFSubset, ScoresDict
from mteb.types.statistics import (
    ImageStatistics,
    LabelStatistics,
    SplitDescriptiveStatistics,
    TextStatistics,
)

from ._statistics_calculation import (
    calculate_image_statistics,
    calculate_label_statistics,
    calculate_text_statistics,
)
from .abstask import AbsTask

logger = logging.getLogger(__name__)


MultilingualDataset = dict[HFSubset, DatasetDict]


def _evaluate_clustering_bootstrapped(
    embeddings: np.ndarray,
    labels: list[list[str]],
    n_clusters: int,
    cluster_size: int,
    kmean_batch_size: int,
    max_depth: int | None,
    rng_state: random.Random,
    seed: int,
) -> tuple[dict[str, list[float]], dict[str, list[list[int]]]]:
    """Bootstrapped evaluation of clustering performance using V-measure.

    The bootstrapping is done by sampling N samples from the corpus and clustering them. It is done without replacement to get a diverse set of
    samples.

    Returns:
        A tuple containing:
        - A dictionary where keys are level names (e.g., "Level 0", "Level 1", etc.) and values are lists of V-measure scores for each clustering experiment at that level.
        - A dictionary where keys are level names and values are lists of cluster assignments for each clustering experiment at that level.
    """
    v_measures = defaultdict(list)
    cluster_assignments = defaultdict(list)
    if max_depth is not None:
        max_depth = min(max_depth, max(map(len, labels)))
    else:
        max_depth = max(map(len, labels))
    # Evaluate on each level til max depth
    for i_level in range(max_depth):
        level_labels = []
        # Assign -1 to gold label if the level is not there
        for label in labels:
            if len(label) > i_level:
                level_labels.append(label[i_level])
            else:
                level_labels.append(-1)
        level_labels = np.array(level_labels)
        valid_idx = np.array(
            [level_label != -1 for level_label in level_labels]
        )  # Could be level_labels != -1 but fails with FutureWarning: elementwise comparison failed
        level_labels = level_labels[valid_idx]
        level_embeddings = embeddings[valid_idx]
        clustering_model = MiniBatchKMeans(
            n_clusters=np.unique(level_labels).size,
            batch_size=kmean_batch_size,
            init="k-means++",
            n_init=1,  # default when kmeans++ is used
            random_state=seed,
        )
        for _ in range(n_clusters):
            # sample N samples from the corpus with replacement
            n_embeddings = len(level_embeddings)
            cluster_indices = rng_state.choices(range(n_embeddings), k=cluster_size)

            _embeddings = level_embeddings[cluster_indices]
            _labels = level_labels[cluster_indices]
            cluster_assignment = clustering_model.fit_predict(_embeddings)
            v_measure = v_measure_score(_labels, cluster_assignment)
            v_measures[f"Level {i_level}"].append(v_measure)
            cluster_assignments[f"Level {i_level}"].append(cluster_assignment.tolist())

    return v_measures, cluster_assignments


class ClusteringFastDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for ClusteringFast

    Attributes:
        num_samples: number of samples in the dataset.

        text_statistics: Statistics for text
        image_statistics: Statistics for images
        labels_statistics: Statistics for labels
    """

    num_samples: int

    text_statistics: TextStatistics | None
    image_statistics: ImageStatistics | None
    labels_statistics: LabelStatistics


class AbsTaskClustering(AbsTask):
    """Abstract class for Clustering tasks.

    This class embeds the corpus sentences then samples N samples from the corpus and clusters them.
    The similarity then is calculated using the V-measure metric, which is invariant to the permutation of the labels.
    This approach is then repeated K times.

    There are two ways to specify how a dataset is downsampled `max_document_to_embed` and `max_fraction_of_documents_to_embed`.
    If both parameters are set to None, no downsampling is done in self._evaluate_subset().
    Only one of these two parameters can be not None at the same time.

    If the clustering is hierarchical, and more than one label is specified in order for each observation,
    V-measures are calculated in the outlined way on each of the levels separately.

    Attributes:
        dataset: A HuggingFace Dataset containing the data for the clustering task. Must contain the following columns `sentences` that contains inputs (texts or images) and labels columns.
        max_fraction_of_documents_to_embed: Fraction of documents to embed for clustering.
        max_document_to_embed: Maximum number of documents to embed for clustering.
        max_documents_per_cluster: Number of documents to sample for each clustering experiment.
        n_clusters: Number of clustering experiments to run.
        k_mean_batch_size: Batch size to use for k-means clustering.
        max_depth: Maximum depth to evaluate clustering. If None, evaluates all levels.
        input_column_name: Name of the column containing the input sentences or data points.
        label_column_name: Name of the column containing the true cluster labels.
        abstask_prompt: Prompt to use for the task for instruction model if not prompt is provided in TaskMetadata.prompt.
    """

    max_fraction_of_documents_to_embed: float | None = 0.04
    max_document_to_embed: int | None = None
    max_documents_per_cluster: int = 16_384
    n_clusters: int = 10
    k_mean_batch_size: int = 512
    max_depth = None
    abstask_prompt = "Identify categories in user passages."
    input_column_name: str = "sentences"
    label_column_name: str = "labels"

    def _evaluate_subset(
        self,
        model: EncoderProtocol,
        data_split: Dataset,
        *,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        **kwargs: Any,
    ) -> ScoresDict:
        if (
            self.max_document_to_embed is not None
            and self.max_fraction_of_documents_to_embed is not None
        ):
            raise Exception(
                "Both max_document_to_embed and max_fraction_of_documents_to_embed are set. Please only set one."
            )

        logger.info("Running clustering - Preparing data...")
        if (
            self.max_document_to_embed is None
            and self.max_fraction_of_documents_to_embed is None
        ):
            downsampled_dataset = data_split
        else:
            if self.max_fraction_of_documents_to_embed is not None:
                max_documents_to_embed = int(
                    self.max_fraction_of_documents_to_embed * len(data_split)
                )
            else:
                max_documents_to_embed = self.max_document_to_embed

            max_documents_to_embed = min(len(data_split), max_documents_to_embed)  # type: ignore
            example_indices = self.rng_state.sample(
                range(len(data_split)), k=max_documents_to_embed
            )
            downsampled_dataset = data_split.select(example_indices)  # type: ignore

        downsampled_dataset = downsampled_dataset.select_columns(
            [self.input_column_name, self.label_column_name]
        )

        logger.info("Running clustering - Encoding samples...")
        embeddings = model.encode(
            create_dataloader(
                downsampled_dataset,
                self.metadata,
                input_column=self.input_column_name,
                **encode_kwargs,
            ),
            task_metadata=self.metadata,
            hf_subset=hf_subset,
            hf_split=hf_split,
            **encode_kwargs,
        )

        logger.info("Running clustering - Evaluating clustering...")
        labels = []
        for label in downsampled_dataset[self.label_column_name]:
            if not isinstance(label, list):
                label = [label]
            labels.append(label)

        all_v_scores, all_assignments = _evaluate_clustering_bootstrapped(
            embeddings,
            labels,
            n_clusters=self.n_clusters,
            cluster_size=self.max_documents_per_cluster,
            kmean_batch_size=self.k_mean_batch_size,
            max_depth=self.max_depth,
            rng_state=self.rng_state,
            seed=self.seed,
        )

        if prediction_folder:
            self._save_task_predictions(
                all_assignments,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        v_measures = list(itertools.chain.from_iterable(all_v_scores.values()))

        logger.info("Running clustering - Finished.")
        mean_v_measure = np.mean(v_measures)
        v_std = np.std(v_measures)
        return {
            "v_measures": all_v_scores,
            "v_measure": float(mean_v_measure),
            "v_measure_std": v_std,
        }

    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> ClusteringFastDescriptiveStatistics:
        if hf_subset:
            inputs = self.dataset[hf_subset][split][self.input_column_name]
            labels = self.dataset[hf_subset][split][self.label_column_name]
        elif compute_overall:
            inputs = []
            labels = []
            for hf_subset in self.metadata.eval_langs:
                inputs.extend(self.dataset[hf_subset][split][self.input_column_name])
                labels.extend(self.dataset[hf_subset][split][self.label_column_name])
        else:
            inputs = self.dataset[split][self.input_column_name]
            labels = self.dataset[split][self.label_column_name]

        if isinstance(inputs[0], list):
            inputs = [item for sublist in inputs for item in sublist]
        if isinstance(labels[0], list):
            labels = [item for sublist in labels for item in sublist]

        text_statistics, image_statistics = None, None
        if "image" in self.metadata.modalities:
            image_statistics = calculate_image_statistics(inputs)

        if "text" in self.metadata.modalities:
            text_statistics = calculate_text_statistics(inputs)

        label_statistics = calculate_label_statistics(labels)

        return ClusteringFastDescriptiveStatistics(
            num_samples=len(inputs),
            text_statistics=text_statistics,
            image_statistics=image_statistics,
            labels_statistics=label_statistics,
        )

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        self._upload_dataset_to_hub(
            repo_name, [self.input_column_name, self.label_column_name]
        )


def _convert_to_fast(
    dataset: DatasetDict,
    input_column_name: str,
    label_column_name: str,
    seed: int,
    max_size: int = 100_000,
) -> DatasetDict:
    """Converts a clustering dataset to a fast version.

    This concat the cluster into two columns, sentences and labels. It additionally downsamples the dataset to max_size.

    Args:
        dataset: A DatasetDict containing the data for the clustering task. Must contain the following columns
        input_column_name: Name of the column containing the input sentences or data points.
        label_column_name: Name of the column containing the true cluster labels.
        seed: Random seed for downsampling.
        max_size: Maximum number of samples in the returned dataset.

    Returns:
        A downsampled DatasetDict with two columns, sentences and labels.
    """
    rng_state = random.Random(seed)

    ds = {}
    for split in dataset:
        sent_set = set()
        labels = []
        sentences = []
        n_clusters = len(dataset[split])
        all_labels_set = set(
            itertools.chain.from_iterable(dataset[split][label_column_name])
        )
        for i in range(n_clusters):
            lab = dataset[split][label_column_name][i]
            sents = dataset[split][input_column_name][i]

            # check that it is the same distribution
            row_label_set = set(lab)
            assert row_label_set.issubset(all_labels_set), (
                "The clusters are not sampled from the same distribution as they have different labels."
            )

            for l, s in zip(lab, sents):
                if s not in sent_set:
                    labels.append(l)
                    sentences.append(s)
                    sent_set.add(s)  # ensuring no duplicates

        ds[split] = Dataset.from_dict(
            {input_column_name: sentences, label_column_name: labels}
        )

        if len(ds[split]) > max_size:
            idxs = rng_state.sample(range(len(ds[split])), max_size)
            ds[split] = ds[split].select(idxs)

    return DatasetDict(ds)


def _check_label_distribution(
    ds: DatasetDict,
    label_column_name: str = "labels",
) -> None:
    """For older clustering dataset versions.

    Checks that all clusters are sampled from the same distribution by checking that the set of labels is the same across clusters.

    Args:
        ds: A DatasetDict containing the data for the clustering task. Must contain the following columns
        label_column_name: Name of the column containing the true cluster labels.
    """
    n_clusters = len(ds)
    if n_clusters > 50:
        return
    all_labels_set = set(itertools.chain.from_iterable(ds[label_column_name]))

    for i in range(n_clusters):
        lab = ds[label_column_name][i]

        # check that it is the same distribution
        row_label_set = set(lab)
        assert row_label_set.issubset(all_labels_set), (
            "The clusters are not sampled from the same distribution as they have different labels."
        )

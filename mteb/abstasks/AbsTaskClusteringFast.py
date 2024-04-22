from __future__ import annotations

import logging
import random
from typing import Any, Dict, List

import numpy as np
import sklearn
import sklearn.cluster
from attr import dataclass
from sklearn.metrics.cluster import v_measure_score

from .AbsTask import AbsTask

logger = logging.getLogger(__name__)

HFLang = str
Split = str
Sentences = List[str]
ColumnName = str
ColumnValues = List[str]
HFDataset = Dict[Split, Dict[ColumnName, ColumnValues]]
MultilingualDataset = Dict[HFLang, HFDataset]
IsoLanguage = str
ClusteringScores = List[Dict[str, float]]
Scores = Dict[str, Any]


def evaluate_clustering_bootstrapped(
    embeddings: np.ndarray,
    labels: list[str],
    n_clusters: int,
    cluster_size: int,
    kmean_batch_size: int,
    rng_state: random.Random = random.Random(),
) -> ClusteringScores:
    """Bootstrapped evaluation of clustering performance using V-measure.

    The bootstrapping is done by sampling N samples from the corpus and clustering them. It is done without replacement to get a diverse set of
    samples.
    """
    n_embeddings = embeddings.shape[0]
    labels_arr = np.array(labels)

    scores = []

    clustering_model = sklearn.cluster.MiniBatchKMeans(
        n_clusters=len(set(labels)),
        batch_size=kmean_batch_size,
        n_init="auto",
    )

    for _ in range(n_clusters):
        # sample N samples from the corpus without replacement
        cluster_indices = rng_state.sample(range(n_embeddings), k=cluster_size)

        _embeddings = embeddings[cluster_indices]
        _labels = labels_arr[cluster_indices]
        clustering_model.fit(_embeddings)
        cluster_assignment = clustering_model.labels_
        v_measure = v_measure_score(_labels, cluster_assignment)
        scores.append({"v_measure": v_measure})

    return scores


class AbsTaskClusteringFast(AbsTask):
    """Abstract class for Clustering tasks.

    This class embeds the corpus sentences then samples N samples from the corpus and clusters them.
    he similarity then is calculated using the V-measure metric, which is invariant to the permutation of the labels.
    This approach is then repeated K times.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset.
    It must contain the following columns:
        sentences: list of str
        labels: list of str
    """

    max_documents_to_embed = 16_384
    max_documents_per_cluster = 2048
    n_clusters = 10
    k_mean_batch_size = 512

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_main_score(self, scores: list[dict[str, float]]) -> float:
        return float(np.mean([score[self.metadata.main_score] for score in scores]))

    def evaluate(
        self, model, split="test", **kwargs
    ) -> (
        dict[HFLang, dict[str, ClusteringScores | float]]
        | dict[str, ClusteringScores | float]
    ):
        self.dataset: MultilingualDataset | HFDataset

        if not self.data_loaded:
            self.load_data()

        if self.is_multilingual:
            multilingual_ds: MultilingualDataset = self.dataset  # type: ignore

            multilingual_scores: dict[HFLang, dict[str, ClusteringScores | float]] = {}
            for lang in self.dataset:
                logger.info(
                    f"\nTask: {self.metadata.name}, split: {split}, language: {lang}. Running..."
                )
                multilingual_scores[lang] = self._evaluate_monolingual(
                    model, multilingual_ds[lang], split, **kwargs
                )
                return multilingual_scores
        logger.info(f"\nTask: {self.metadata.name}, split: {split}. Running...")

        ds: HFDataset = self.dataset  # type: ignore
        scores = self._evaluate_monolingual(model, ds, split, **kwargs)
        return scores

    def _evaluate_monolingual(
        self, model, dataset: HFDataset, split: str = "test", **kwargs: Any
    ) -> dict[str, float | ClusteringScores]:
        sentences = dataset[split]["sentences"][: self.max_documents_to_embed]
        labels = dataset[split]["labels"][: self.max_documents_to_embed]

        logger.info(f"Encoding {len(labels)} sentences...")

        embeddings = model.encode(sentences)  # , prompt_name=self.metadata.name)

        scores = evaluate_clustering_bootstrapped(
            embeddings,
            labels,
            n_clusters=self.n_clusters,
            cluster_size=self.max_documents_per_cluster,
            kmean_batch_size=self.k_mean_batch_size,
            rng_state=random.Random(self.seed),
        )

        return {"main_score": self.calculate_main_score(scores), "scores": scores}


@dataclass
class MTEBScores:
    """The MTEB score object.

    Attributes:
        task_name: The name of the task
        task_revision: The revision of the task
        mteb_version: The version of MTEB used
        main_score_name: The name of the main score (e.g. "Accuracy")
        scores: The scores for each language. The keys are the 3 letter languages codes (ISO 639-3) with a 4 letter script code (ISO 15924).
            For example, English in Latin script is "eng_Latn". The values are dictionaries with the score names as keys and the scores as values.
        evaluation_time: The time in seconds it took to evaluate the task
        main_score: The main score for each language
    """

    # task metadata
    task_name: str
    task_revision: str
    mteb_version: str

    # scores
    main_score_name: str
    scores: dict[IsoLanguage, dict[Split, Scores]]
    evaluation_time: float
    main_score: dict[IsoLanguage, dict[Split, float]]

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "task_revision": self.task_revision,
            "mteb_version": self.mteb_version,
            "main_score_name": self.main_score_name,
            "scores": self.scores,
            "evaluation_time": self.evaluation_time,
            "main_score": self.main_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MTEBScores:
        return cls(
            task_name=data["task_name"],
            task_revision=data["task_revision"],
            mteb_version=data["mteb_version"],
            main_score_name=data["main_score_name"],
            scores=data["scores"],
            evaluation_time=data["evaluation_time"],
            main_score=data["main_score"],
        )

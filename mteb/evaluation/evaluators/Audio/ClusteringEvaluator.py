from __future__ import annotations

import logging
import random
from typing import Any

import sklearn
import sklearn.cluster
from datasets import Audio
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.decomposition import PCA

from mteb.encoder_interface import Encoder
from mteb.evaluation.evaluators.Evaluator import Evaluator

logger = logging.getLogger(__name__)


class AudioClusteringEvaluator(Evaluator):
    def __init__(
        self,
        audio: list[Audio],
        labels: list[int],
        task_name: str | None = None,
        clustering_batch_size: int = 500,
        limit: int | None = None,
        cluster_algo: str = "KMeans",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit is not None:
            audio = audio[:limit]
            labels = labels[:limit]

        random.seed(42)
        combined = list(zip(audio, labels))
        random.shuffle(combined)
        audio, labels = map(list, zip(*combined))

        self.audio = audio
        self.labels = labels
        self.clustering_batch_size = clustering_batch_size
        self.task_name = task_name
        self.cluster_algo = cluster_algo

    def __clustering__(self):
        if self.cluster_algo == "Kmeans":
            logger.info("Fitting Mini-Batch K-Means model...")
            clustering_model = sklearn.cluster.MiniBatchKMeans(
                n_clusters=len(set(self.labels)),
                batch_size=self.clustering_batch_size,
                n_init="auto",
            )
        elif self.cluster_algo == "DBSCAN":
            # need to plot out the distribution of the embeddings to decide on parameters for DBSCAN
            logger.info("Fitting DBSCAN model...")
            clustering_model = sklearn.cluster.DBSCAN(
                eps=0.5, min_samples=5, metric="euclidean"
            )
        elif self.cluster_algo == "Agg":
            logger.info("Fitting Agglomerative model...")
            clustering_model = sklearn.cluster.AgglomerativeClustering(
                n_clusters=len(set(self.labels)), linkage="average", metric="cosine"
            )
        return clustering_model

    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 32

        audio_embeddings = model.get_audio_embeddings(
            self.audio,
            batch_size=encode_kwargs["batch_size"],
        )

        logger.info("Fitting Mini-Batch K-Means model...")

        pca = PCA(n_components=200)
        audio_embeddings = pca.fit_transform(audio_embeddings)

        clustering_output = self.__clustering__()
        clustering_output.fit(audio_embeddings)
        cluster_assignment = clustering_output.labels_

        logger.info("Evaluating...")
        v_measure = metrics.cluster.v_measure_score(self.labels, cluster_assignment)
        nmi = metrics.cluster.normalized_mutual_info_score(
            self.labels, cluster_assignment
        )
        ari = metrics.cluster.adjusted_rand_score(self.labels, cluster_assignment)

        matrix = metrics.confusion_matrix(self.labels, cluster_assignment)

        silhouette = float(
            metrics.silhouette_score(
                audio_embeddings, cluster_assignment, metric="euclidean"
            )
        )
        print(self.cluster_algo)
        # get linear sum assignment
        row_ind, col_ind = linear_sum_assignment(matrix, maximize=True)
        total_correct = matrix[row_ind, col_ind].sum()
        clustering_accuracy = total_correct / len(self.labels)

        return {
            "v_measure": v_measure,
            "nmi": nmi,
            "ari": ari,
            "cluster_accuracy": clustering_accuracy,
            "silhouette": silhouette,
        }

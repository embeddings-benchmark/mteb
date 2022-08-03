import logging
import random
import multiprocessing

import numpy as np
import sklearn
import sklearn.cluster


logger = logging.getLogger(__name__)

from .Evaluator import Evaluator


class ClusteringEvaluator(Evaluator):
    def __init__(self, sentences, labels, clustering_batch_size=500, limit=None, **kwargs):
        if limit is not None:
            sentences = sentences[:limit]
            labels = labels[:limit]
        self.sentences = sentences
        self.labels = labels

        # Set seed since KMeans is used
        seed = 28042000
        random.seed(seed)
        np.random.seed(seed)

        self.clustering_batch_size = multiprocessing.cpu_count() * 256#clustering_batch_size
        print("Using BS ", self.clustering_batch_size)

    def __call__(self, model):
        logger.info(f"Encoding {len(self.sentences)} sentences...")
        corpus_embeddings = np.asarray(model.encode(self.sentences))

        print("Got embeddings of shape ", corpus_embeddings.shape, len(set(self.labels)))
        logger.info("Fitting Mini-Batch K-Means model...")
        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=2, batch_size=self.clustering_batch_size
        )
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        logger.info("Evaluating...")
        v_measure = sklearn.metrics.cluster.v_measure_score(self.labels, cluster_assignment)

        return {"v_measure": v_measure}

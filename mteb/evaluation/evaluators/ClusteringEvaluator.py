from .Evaluator import Evaluator
import random
import numpy as np
import sklearn
import sklearn.cluster


class ClusteringEvaluator(Evaluator):
    def __init__(self, sentences, labels, batch_size=500, limit=None, **kwargs):
        if limit is not None:
            sentences = sentences[:limit]
            labels = labels[:limit]
        self.sentences = sentences
        self.labels = labels

        # Set seed since KMeans is used
        seed = 28042000
        random.seed(seed)
        np.random.seed(seed)

        self.batch_size = batch_size

    def __call__(self, model):
        corpus_embeddings = np.asarray(model.encode(self.sentences))

        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=len(set(self.labels)), batch_size=self.batch_size
        )
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        v_measure = sklearn.metrics.cluster.v_measure_score(self.labels, cluster_assignment)

        return {"v_measure": v_measure}

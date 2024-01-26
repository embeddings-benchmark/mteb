import logging
from mteb.utils import get_embed_with_lang_func

import numpy as np
import sklearn
import sklearn.cluster

logger = logging.getLogger(__name__)

from .Evaluator import Evaluator


class ClusteringEvaluator(Evaluator):
    def __init__(self, sentences, labels, language, clustering_batch_size=500, batch_size=32, limit=None, **kwargs):
        super().__init__(**kwargs)
        if limit is not None:
            sentences = sentences[:limit]
            labels = labels[:limit]
        self.sentences = sentences
        self.labels = labels
        self.clustering_batch_size = clustering_batch_size
        self.batch_size = batch_size
        self.language = language

    def __call__(self, model):
        logger.info(f"Encoding {len(self.sentences)} sentences...")
        embed_fn = get_embed_with_lang_func(model)
        corpus_embeddings = np.asarray(embed_fn(self.sentences, batch_size=self.batch_size, language=self.language))

        logger.info("Fitting Mini-Batch K-Means model...")
        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=len(set(self.labels)), batch_size=self.clustering_batch_size, n_init="auto"
        )
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        logger.info("Evaluating...")
        v_measure = sklearn.metrics.cluster.v_measure_score(self.labels, cluster_assignment)

        return {"v_measure": v_measure}

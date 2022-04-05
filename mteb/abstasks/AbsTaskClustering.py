from AbsTask import AbsTask
from datasets import load_dataset
import os
import numpy as np
import gzip
import json
import sklearn
import sklearn.cluster
import tqdm
import random
import numpy as np

class AbsTaskClustering(AbsTask):
    def __init__(self, task_json):
        super(AbsTaskClustering, self).__init__()
        self.task_json = task_json
        self.dataset = None
        self.data_loaded = False
        self.clustering_sets = None

    def load_data(self):
        if self.data_loaded:
            return
        self.dataset = load_dataset(self.task_json['hf_name'])
        self.data_loaded = True

    def evaluate(self, model, split='test'):
        if not self.data_loaded:
            self.load_data()

        random.seed(42)
        np.random.seed(42)

        v_measures = []
        for cluster_set in tqdm.tqdm(self.clustering_sets[split], desc='Clustering'):
            v_measures.append(self.eval_clustering(model, cluster_set['sentences'], cluster_set['labels']))

        v_mean = np.mean(v_measures)
        v_std = np.std(v_measures)
        return {'v_measure': v_mean, 'v_measure_std': v_std}

    def eval_clustering(self, model, sentences, labels):
        corpus_embeddings = np.asarray(model.encode(sentences, show_progress_bar=False))
        clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=len(set(labels)), batch_size=2000)

        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        return sklearn.metrics.cluster.v_measure_score(labels, cluster_assignment)
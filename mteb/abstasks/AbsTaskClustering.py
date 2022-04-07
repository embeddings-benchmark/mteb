from .AbsTask import AbsTask
import datasets
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
    def __init__(self):
        super(AbsTaskClustering, self).__init__()
        self.dataset = None
        self.data_loaded = False

    def load_data(self):
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(self.description['hf_hub_name'])
        self.data_loaded = True

    def evaluate(self, model, split='test'):
        if not self.data_loaded:
            self.load_data()

        v_measures = []
        for cluster_set in tqdm.tqdm(self.dataset[split], desc='Clustering'):
            v_measures.append(self.eval_clustering(model, cluster_set['sentences'], cluster_set['labels']))

        v_mean = np.mean(v_measures)
        v_std = np.std(v_measures)
        return {'v_measure': v_mean, 'v_measure_std': v_std}

    def eval_clustering(self, model, sentences, labels):
        corpus_embeddings = np.asarray(model.encode(sentences))
        clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=len(set(labels)), batch_size=2000)

        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        return sklearn.metrics.cluster.v_measure_score(labels, cluster_assignment)
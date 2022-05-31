from .AbsTask import AbsTask
from ..evaluation.evaluators import ClusteringEvaluator
import datasets
import numpy as np
import tqdm
import random
import numpy as np


class AbsTaskClustering(AbsTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed = 42

    def evaluate(self, model, split="test"):
        if not self.data_loaded:
            self.load_data()

        v_measures = []
        for cluster_set in tqdm.tqdm(self.dataset[split], desc="Clustering"):
            evaluator = ClusteringEvaluator(cluster_set["sentences"], cluster_set["labels"])
            metrics = evaluator(model)
            v_measures.append(metrics["v_measure"])

        v_mean = np.mean(v_measures)
        v_std = np.std(v_measures)
        return {"v_measure": v_mean, "v_measure_std": v_std}

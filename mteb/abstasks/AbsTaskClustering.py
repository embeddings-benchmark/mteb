import numpy as np
import tqdm

from ..evaluation.evaluators import ClusteringEvaluator
from .AbsTask import AbsTask


class AbsTaskClustering(AbsTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed = 42

    def evaluate(self, model, split="test", **kwargs):
        if not self.data_loaded:
            self.load_data()

        v_measures = []
        print("GOTDATA", self.dataset[split])
        for cluster_set in tqdm.tqdm(self.dataset[split], desc="Clustering"):
            print("LEN", len(cluster_set))
            evaluator = ClusteringEvaluator(cluster_set["sentences"], cluster_set["labels"], **kwargs)
            metrics = evaluator(model)
            v_measures.append(metrics["v_measure"])

        v_mean = np.mean(v_measures)
        v_std = np.std(v_measures)
        return {"v_measure": v_mean, "v_measure_std": v_std}

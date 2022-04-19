from .AbsTask import AbsTask
import datasets
from sentence_transformers import evaluation
import numpy as np
import logging
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np

class AbsTaskSTS(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = {'dev': Dict[id, str], 'test': Dict[id, str]}         #id => sentence
    self.queries = {'dev': Dict[id, str], 'test': Dict[id, str]}
    self.relevant_docs = {'dev': Dict[id, set], 'test': Dict[id, set]}
    """
    def __init__(self, **kwargs):
        super(AbsTaskSTS, self).__init__(**kwargs)        
        self.dataset = None
        self.data_loaded = False

    @property
    def min_score(self):
        return self.description['min_score']

    @property
    def max_score(self):
        return self.description['max_score']

    def load_data(self):
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(self.description['hf_hub_name'])
        self.data_loaded = True

    def evaluate(self, model, split):
        if not self.data_loaded:
            self.load_data()

        data_split = self.dataset[split]

        embeddings1 = np.asarray(model.encode(data_split['sentence1']))
        embeddings2 = np.asarray(model.encode(data_split['sentence2']))

        gold_scores = data_split['score']

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)

        cosine_pearson, _ = pearsonr(gold_scores, cosine_scores)
        cosine_spearman, _ = spearmanr(gold_scores, cosine_scores)

        manhatten_pearson, _ = pearsonr(gold_scores, manhattan_distances)
        manhatten_spearman, _ = spearmanr(gold_scores, manhattan_distances)

        euclidean_pearson, _ = pearsonr(gold_scores, euclidean_distances)
        euclidean_spearman, _ = spearmanr(gold_scores, euclidean_distances)

        return {
            'cosine_pearson': cosine_pearson,
            'cosine_spearman': cosine_spearman,
            'manhatten_pearson': manhatten_pearson,
            'manhatten_spearman': manhatten_spearman,
            'euclidean_pearson': euclidean_pearson,
            'euclidean_spearman': euclidean_spearman,
        }
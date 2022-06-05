from .Evaluator import Evaluator
import numpy as np
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)
from scipy.stats import pearsonr, spearmanr


class STSEvaluator(Evaluator):
    def __init__(self, sentences1, sentences2, gold_scores, limit=None):
        if limit is not None:
            sentences1 = sentences1[:limit]
            sentences2 = sentences2[:limit]
            gold_scores = gold_scores[:limit]
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.gold_scores = gold_scores

    def __call__(self, model):
        embeddings1 = np.asarray(model.encode(self.sentences1))
        embeddings2 = np.asarray(model.encode(self.sentences2))

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)

        cosine_pearson, _ = pearsonr(self.gold_scores, cosine_scores)
        cosine_spearman, _ = spearmanr(self.gold_scores, cosine_scores)

        manhatten_pearson, _ = pearsonr(self.gold_scores, manhattan_distances)
        manhatten_spearman, _ = spearmanr(self.gold_scores, manhattan_distances)

        euclidean_pearson, _ = pearsonr(self.gold_scores, euclidean_distances)
        euclidean_spearman, _ = spearmanr(self.gold_scores, euclidean_distances)

        return {
            "cosine_pearson": cosine_pearson,
            "cosine_spearman": cosine_spearman,
            "manhatten_pearson": manhatten_pearson,
            "manhatten_spearman": manhatten_spearman,
            "euclidean_pearson": euclidean_pearson,
            "euclidean_spearman": euclidean_spearman,
        }

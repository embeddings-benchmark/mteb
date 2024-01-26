import logging
from mteb.utils import get_embed_with_lang_func, maybe_split_language_pair

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

logger = logging.getLogger(__name__)

from .Evaluator import Evaluator


class STSEvaluator(Evaluator):
    def __init__(self, sentences1, sentences2, gold_scores, language, batch_size=64, limit=None, **kwargs):
        super().__init__(**kwargs)
        if limit is not None:
            sentences1 = sentences1[:limit]
            sentences2 = sentences2[:limit]
            gold_scores = gold_scores[:limit]
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.gold_scores = gold_scores
        self.batch_size = batch_size
        self.language = language

    def __call__(self, model):
        logger.info(f"Encoding {len(self.sentences1)} sentences1...")
        embed_fn = get_embed_with_lang_func(model)
        lang1, lang2 = maybe_split_language_pair(self.language)
        embeddings1 = np.asarray(embed_fn(self.sentences1, batch_size=self.batch_size, language=lang1))
        logger.info(f"Encoding {len(self.sentences2)} sentences2...")
        embeddings2 = np.asarray(embed_fn(self.sentences2, batch_size=self.batch_size, language=lang2))

        logger.info("Evaluating...")
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
            "cos_sim": {
                "pearson": cosine_pearson,
                "spearman": cosine_spearman,
            },
            "manhattan": {
                "pearson": manhatten_pearson,
                "spearman": manhatten_spearman,
            },
            "euclidean": {
                "pearson": euclidean_pearson,
                "spearman": euclidean_spearman,
            },
        }

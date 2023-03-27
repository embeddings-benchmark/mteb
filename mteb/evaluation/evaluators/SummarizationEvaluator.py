import logging

import numpy as np
import torch
from tqdm import trange

from scipy.stats import pearsonr, spearmanr

from .utils import cos_sim, dot_score


logger = logging.getLogger(__name__)

from .Evaluator import Evaluator


class SummarizationEvaluator(Evaluator):
    def __init__(
        self, human_summaries=None, machine_summaries=None, texts=None, gold_scores=None, limit=None, **kwargs
    ):
        # human_summaries shape: (None, num_human_summaries)
        # machine_summaries shape: (None, num_machine_summaries)
        # gold scores shape: (None, num_machine_summaries)
        # texts: (None,)
        super().__init__(**kwargs)
        if limit is not None:
            human_summaries = human_summaries[:limit]
            machine_summaries = machine_summaries[:limit]
            gold_scores = gold_scores[:limit]
            texts = texts[:limit]
        self.human_summaries = human_summaries
        self.machine_summaries = machine_summaries
        self.texts = texts
        self.gold_scores = gold_scores

    def __call__(self, model):

        cosine_spearman_scores = []
        cosine_pearson_scores = []
        dot_spearman_scores = []
        dot_pearson_scores = []

        for i in trange(len(self.texts), desc="Texts"):  # iterate over all original texts
            # Get the human & machine summaries for the text
            embs_human_summaries = model.encode(self.human_summaries[i])
            embs_machine_summaries = model.encode(self.machine_summaries[i])

            cosine_pred_scores = []  # Predicted quality score for a summary
            dot_pred_scores = []  # Predicted quality score for a summary
            human_scores = []  # Human score for a summary
            for emb_machine_summary, human_eval_score in zip(
                embs_machine_summaries, self.gold_scores[i]
            ):
                cosine_scores = cos_sim(emb_machine_summary, embs_human_summaries)
                dot_scores = dot_score(emb_machine_summary, embs_human_summaries)

                cosine_max_score = torch.max(cosine_scores).item()
                cosine_pred_scores.append(cosine_max_score)
                dot_max_score = torch.max(dot_scores).item()
                dot_pred_scores.append(dot_max_score)
                human_scores.append(human_eval_score)

            if (len(set(human_scores)) == 1) or (len(set(dot_pred_scores)) == 1) or (len(set(cosine_pred_scores)) == 1):
                logger.info(f"Skipping sample {i} due to equal scores")
                continue

            cosine_spearman_scores.append(spearmanr(human_scores, cosine_pred_scores))
            cosine_pearson_scores.append(pearsonr(human_scores, cosine_pred_scores))
            dot_spearman_scores.append(spearmanr(human_scores, dot_pred_scores))
            dot_pearson_scores.append(pearsonr(human_scores, dot_pred_scores))

        cosine_spearman = np.mean(cosine_spearman_scores)
        dot_spearman = np.mean(dot_spearman_scores)
        cosine_pearson = np.mean(cosine_pearson_scores)
        dot_pearson = np.mean(dot_pearson_scores)

        return {
            "cos_sim": {
                "spearman": cosine_spearman,
                "pearson": cosine_pearson,
            },
            "dot": {
                "spearman": dot_spearman,
                "pearson": dot_pearson,
            },
        }

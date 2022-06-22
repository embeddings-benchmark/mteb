import logging

import numpy as np
import torch

from scipy.stats import pearsonr, spearmanr

from .utils import cos_sim


logger = logging.getLogger(__name__)

from .Evaluator import Evaluator


class SummarizationEvaluator(Evaluator):
    def __init__(
        self, human_summaries=None, machine_summaries=None, texts=None, gold_scores=None, limit=None, **kwargs
    ):
        # human_summaries shape : (None, num_human_summaries)
        # machine_summaries shape : (None, num_machine_summaries)
        # gold scores shape : (None, num_machine_summaries)
        # texts: (None,)

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

        all_spearman_corr_scores = []

        for i in range(len(self.texts)):  # iterate over all original texts
            human_summaries = self.human_summaries[i]  # Get the human summaries for the text
            embs_human_summaries = model.encode(human_summaries)
            pred_scores = []  # Our predict quality score for a summary
            human_scores = []  # Our human score for a summary
            for machine_summary, human_eval_score in zip(
                self.machine_summaries[i], self.gold_scores[i]
            ):  # Get all machine summaries + scores for this text
                emb_machine_summary = model.encode(machine_summary)  # 1 embedding for the summary
                scores = cos_sim(emb_machine_summary, embs_human_summaries)
                max_score = torch.max(scores).item()
                pred_scores.append(max_score)
                human_scores.append(human_eval_score)

            all_spearman_corr_scores.append(spearmanr(human_scores, pred_scores))

        cosine_spearman = np.mean(all_spearman_corr_scores)

        return {
            "cos_sim": {
                "spearman": cosine_spearman,
            },
        }

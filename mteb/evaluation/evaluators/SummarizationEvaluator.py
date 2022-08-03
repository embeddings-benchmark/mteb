import logging
from sys import prefix

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
            human_summaries = self.human_summaries[i]  # Get the human summaries for the text
            embs_human_summaries = model.encode(human_summaries)
            cosine_pred_scores = []  # Predicted quality score for a summary
            dot_pred_scores = []  # Predicted quality score for a summary
            human_scores = []  # Human score for a summary
            for machine_summary, human_eval_score in zip(
                self.machine_summaries[i], self.gold_scores[i]
            ):  # Get all machine summaries + scores for this text
                emb_machine_summary = model.encode(
                    machine_summary, show_progress_bar=False
                )  # 1 embedding for the summary
                cosine_scores = cos_sim(emb_machine_summary, embs_human_summaries)
                dot_scores = dot_score(emb_machine_summary, embs_human_summaries)

                cosine_max_score = torch.max(cosine_scores).item()
                cosine_pred_scores.append(cosine_max_score)
                dot_max_score = torch.max(dot_scores).item()
                dot_pred_scores.append(dot_max_score)
                human_scores.append(human_eval_score)

            if (len(set(human_scores)) == 1) or (len(set(dot_pred_scores)) == 1) or len(set(cosine_pred_scores)) == 1:
                logging.info(f"Skipping sample {i} due to equal scores")
                continue
            #if len(set(human_scores)) == 1:
            #    print("SAME HUMAN", i, human_scores, human_summaries, self.machine_summaries[i])
            #if len(set(dot_pred_scores)) == 1:
            #    print("SAME DOT", i, dot_pred_scores, human_summaries, self.machine_summaries[i])
            #if len(set(cosine_pred_scores)) == 1:
            #    print("SAME COSINE", i, cosine_pred_scores, human_summaries, self.machine_summaries[i])

            #if (np.all(human_scores == human_scores[0])) or \
            #    (np.all(cosine_pred_scores == cosine_pred_scores[0])) or \
            #    (np.all(dot_pred_scores == dot_pred_scores[0])):
            #    print("GOT ALL THE SAME")
            #    print(np.all(human_scores == human_scores[0]))
            #    print(np.all(cosine_pred_scores == cosine_pred_scores[0]))
            #    print(np.all(dot_pred_scores == dot_pred_scores[0]))
            #    print(i)


            cosine_spearman_scores.append(spearmanr(human_scores, cosine_pred_scores))
            cosine_pearson_scores.append(pearsonr(human_scores, cosine_pred_scores))
            dot_spearman_scores.append(spearmanr(human_scores, dot_pred_scores))
            dot_pearson_scores.append(pearsonr(human_scores, dot_pred_scores))

        cosine_spearman = np.mean(np.array(cosine_spearman_scores)[~np.isnan(cosine_spearman_scores)])
        dot_spearman = np.mean(np.array(dot_spearman_scores)[~np.isnan(dot_spearman_scores)])
        cosine_pearson = np.mean(np.array(cosine_pearson_scores)[~np.isnan(cosine_pearson_scores)])
        dot_pearson = np.mean(np.array(dot_pearson_scores)[~np.isnan(dot_pearson_scores)])

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

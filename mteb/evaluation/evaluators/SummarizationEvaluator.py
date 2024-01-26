import logging
from mteb.utils import get_embed_with_lang_func

import numpy as np
import torch
import tqdm
from scipy.stats import pearsonr, spearmanr

from .utils import cos_sim, dot_score

logger = logging.getLogger(__name__)

from .Evaluator import Evaluator


class SummarizationEvaluator(Evaluator):
    def __init__(
        self,
        human_summaries=None,
        machine_summaries=None,
        texts=None,
        language=None,
        gold_scores=None,
        limit=None,
        batch_size=32,
        **kwargs
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
        self.batch_size = batch_size
        self.language = language

    def __call__(self, model):
        cosine_spearman_scores = []
        cosine_pearson_scores = []
        dot_spearman_scores = []
        dot_pearson_scores = []

        # Get the human & machine summaries for the text in one go for all
        human_lens = [len(human_summaries) for human_summaries in self.human_summaries]
        machine_lens = [len(machine_summaries) for machine_summaries in self.machine_summaries]

        logger.info(f"Encoding {sum(human_lens)} human summaries...")
        embed_fn = get_embed_with_lang_func(model)
        embs_human_summaries_all = embed_fn(
            [summary for human_summaries in self.human_summaries for summary in human_summaries],
            batch_size=self.batch_size,
            language=self.language,
        )
        logger.info(f"Encoding {sum(machine_lens)} machine summaries...")
        embs_machine_summaries_all = embed_fn(
            [summary for machine_summaries in self.machine_summaries for summary in machine_summaries],
            batch_size=self.batch_size,
            language=self.language,
        )

        # Split the embeddings into the original human & machine summaries
        embs_human_summaries_all = np.split(embs_human_summaries_all, np.cumsum(human_lens)[:-1])
        embs_machine_summaries_all = np.split(embs_machine_summaries_all, np.cumsum(machine_lens)[:-1])

        for i, (embs_human_summaries, embs_machine_summaries) in tqdm.tqdm(
            enumerate(zip(embs_human_summaries_all, embs_machine_summaries_all)),
            desc="Scoring",
            total=len(self.human_summaries),
        ):
            cosine_pred_scores = []  # Predicted quality score for a summary
            dot_pred_scores = []  # Predicted quality score for a summary
            human_scores = []  # Human score for a summary
            for emb_machine_summary, human_eval_score in zip(
                embs_machine_summaries, self.gold_scores[i]
            ):  # Iterate through all machine summaries + scores for a single sample
                cosine_scores = cos_sim(emb_machine_summary, embs_human_summaries)
                dot_scores = dot_score(emb_machine_summary, embs_human_summaries)

                cosine_max_score = torch.max(cosine_scores).item()
                cosine_pred_scores.append(cosine_max_score)
                dot_max_score = torch.max(dot_scores).item()
                dot_pred_scores.append(dot_max_score)
                human_scores.append(human_eval_score)

            if (
                (len(set(human_scores)) == 1)
                or (len(set(dot_pred_scores)) == 1)
                or (len(set(cosine_pred_scores)) == 1)
            ):
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

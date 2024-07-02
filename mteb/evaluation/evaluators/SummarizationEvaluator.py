from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import tqdm
from scipy.stats import pearsonr, spearmanr

from mteb.encoder_interface import Encoder, EncoderWithSimilarity

from .Evaluator import Evaluator
from .model_encode import model_encode
from .utils import cos_sim, dot_score

logger = logging.getLogger(__name__)


class SummarizationEvaluator(Evaluator):
    def __init__(
        self,
        task_name: str | None = None,
        human_summaries=None,
        machine_summaries=None,
        texts=None,
        gold_scores=None,
        limit: int | None = None,
        **kwargs,
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
        self.task_name = task_name

    def __call__(
        self,
        model: Encoder | EncoderWithSimilarity,
        *,
        encode_kwargs: dict[str, Any] = {},
    ):
        # set default for encode_kwargs
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 32

        cosine_spearman_scores = []
        cosine_pearson_scores = []
        dot_spearman_scores = []
        dot_pearson_scores = []
        pearson_scores = []
        spearman_scores = []

        # Get the human & machine summaries for the text in one go for all
        human_lens = [len(human_summaries) for human_summaries in self.human_summaries]
        machine_lens = [
            len(machine_summaries) for machine_summaries in self.machine_summaries
        ]

        logger.info("Encoding human summaries...")
        embs_human_summaries_all = model_encode(
            [
                summary
                for human_summaries in self.human_summaries
                for summary in human_summaries
            ],
            model=model,
            prompt_name=self.task_name,
            **encode_kwargs,
        )

        logger.info("Encoding machine summaries...")
        embs_machine_summaries_all = model_encode(
            [
                summary
                for machine_summaries in self.machine_summaries
                for summary in machine_summaries
            ],
            model=model,
            prompt_name=self.task_name,
            **encode_kwargs,
        )

        # Split the embeddings into the original human & machine summaries
        embs_human_summaries_all = np.split(
            embs_human_summaries_all, np.cumsum(human_lens)[:-1]
        )
        embs_machine_summaries_all = np.split(
            embs_machine_summaries_all, np.cumsum(machine_lens)[:-1]
        )

        for i, (embs_human_summaries, embs_machine_summaries) in tqdm.tqdm(
            enumerate(zip(embs_human_summaries_all, embs_machine_summaries_all)),
            desc="Scoring",
            total=len(self.human_summaries),
        ):
            cosine_pred_scores = []  # Predicted quality score for a summary
            dot_pred_scores = []  # Predicted quality score for a summary
            sim_scores = []
            human_scores = []  # Human score for a summary

            for emb_machine_summary, human_eval_score in zip(
                embs_machine_summaries, self.gold_scores[i]
            ):  # Iterate through all machine summaries + scores for a single sample
                cosine_scores = cos_sim(emb_machine_summary, embs_human_summaries)
                dot_scores = dot_score(emb_machine_summary, embs_human_summaries)

                if hasattr(model, "similarity_pairwise"):
                    # Pairwise similarity
                    _sim_score = [
                        float(model.similarity(emb_machine_summary, emb_human_summary))  # type: ignore
                        for emb_human_summary in embs_human_summaries
                    ]
                    sim_score = torch.tensor(_sim_score)
                elif hasattr(model, "similarity"):
                    _sim_score = [
                        float(model.similarity(emb_machine_summary, emb_human_summary))  # type: ignore
                        for emb_human_summary in embs_human_summaries
                    ]
                    sim_score = torch.tensor(_sim_score)
                else:
                    sim_score = cosine_scores  # Default to cosine similarity

                cosine_max_score = torch.max(cosine_scores).item()
                dot_max_score = torch.max(dot_scores).item()
                sim_max_score = torch.max(sim_score).item()

                cosine_pred_scores.append(cosine_max_score)
                dot_pred_scores.append(dot_max_score)
                sim_scores.append(sim_max_score)
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
            spearman_scores.append(spearmanr(human_scores, sim_scores))
            pearson_scores.append(pearsonr(human_scores, sim_scores))

        return {
            "pearson": np.mean(pearson_scores),
            "spearman": np.mean(spearman_scores),
            "cosine_spearman": np.mean(cosine_spearman_scores),
            "cosine_pearson": np.mean(cosine_pearson_scores),
            "dot_spearman": np.mean(dot_spearman_scores),
            "dot_pearson": np.mean(dot_pearson_scores),
        }

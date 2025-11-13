import logging
import sys
from typing import Any, TypedDict

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

from mteb._create_dataloaders import _create_dataloader_from_texts
from mteb._evaluators.evaluator import Evaluator
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models import EncoderProtocol
from mteb.similarity_functions import cos_sim, dot_score

# if later than python 3.13 use typing module
if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

logger = logging.getLogger(__name__)


class SummarizationDistances(TypedDict):
    """Summarization distances.

    Attributes:
        cosine_scores: Cosine similarity scores.
        similarity_scores: Similarity scores.
        dot_scores: Dot similarity scores.
    """

    cosine_scores: list[list[float]]
    similarity_scores: list[list[float]]
    dot_scores: list[list[float]]
    human_scores: list[list[float]]


class SummarizationMetrics(TypedDict):
    """Summarization metrics.

    Attributes:
        pearson: Pearson correlation coefficient.
        spearman: Spearman correlation
        cosine_spearman: Spearman correlation
        cosine_pearson: Pearson correlation coefficient.
        dot_spearman: Spearman correlation
        dot_pearson: Pearson correlation coefficient.
    """

    pearson: float
    spearman: float
    cosine_spearman: float
    cosine_pearson: float
    dot_spearman: float
    dot_pearson: float


class SummarizationEvaluator(Evaluator):
    def __init__(
        self,
        human_summaries: list[list[str]],
        machine_summaries: list[list[str]],
        texts: list[str],
        gold_scores: list[list[float]],
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        **kwargs,
    ) -> None:
        """Summarization Evaluator

        Args:
        human_summaries: shape: (-1, num_human_summaries)
        machine_summaries: shape: (-1, num_machine_summaries)
        texts: shape: (-1,)
        gold_scores: shape: (-1, num_machine_summaries)
        task_metadata: Name of the task
        hf_split: Split of task
        hf_subset: Subset of task
        **kwargs: Additional arguments to pass to the Evaluator
        """
        super().__init__(**kwargs)
        self.human_summaries = human_summaries
        self.machine_summaries = machine_summaries
        self.texts = texts
        self.gold_scores = gold_scores
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

    def __call__(
        self,
        model: EncoderProtocol,
        *,
        encode_kwargs: dict[str, Any],
    ) -> SummarizationDistances:
        # Get the human & machine summaries for the text in one go for all
        human_lens = [len(human_summaries) for human_summaries in self.human_summaries]
        machine_lens = [
            len(machine_summaries) for machine_summaries in self.machine_summaries
        ]

        logger.info("Encoding human summaries...")
        embs_human_summaries_all = model.encode(
            _create_dataloader_from_texts(
                [
                    summary
                    for human_summaries in self.human_summaries
                    for summary in human_summaries
                ],
                **encode_kwargs,
            ),
            task_metadata=self.task_metadata,
            hf_subset=self.hf_subset,
            hf_split=self.hf_split,
            **encode_kwargs,
        )

        logger.info("Encoding machine summaries...")
        embs_machine_summaries_all = model.encode(
            _create_dataloader_from_texts(
                [
                    summary
                    for machine_summaries in self.machine_summaries
                    for summary in machine_summaries
                ],
                **encode_kwargs,
            ),
            task_metadata=self.task_metadata,
            hf_subset=self.hf_subset,
            hf_split=self.hf_split,
            **encode_kwargs,
        )

        # Split the embeddings into the original human & machine summaries
        embs_human_summaries_all = np.split(
            embs_human_summaries_all, np.cumsum(human_lens)[:-1]
        )
        embs_machine_summaries_all = np.split(
            embs_machine_summaries_all, np.cumsum(machine_lens)[:-1]
        )

        all_cosine_scores = []
        all_dot_scores = []
        all_sim_scores = []
        all_human_scores = []

        for i, (embs_human_summaries, embs_machine_summaries) in tqdm(
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

                _sim_score = [
                    float(model.similarity(emb_machine_summary, emb_human_summary))  # type: ignore
                    for emb_human_summary in embs_human_summaries
                ]
                sim_score = torch.tensor(_sim_score)

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

            all_cosine_scores.append(cosine_pred_scores)
            all_dot_scores.append(dot_pred_scores)
            all_sim_scores.append(sim_scores)
            all_human_scores.append(human_scores)
        return SummarizationDistances(
            cosine_scores=all_cosine_scores,
            dot_scores=all_dot_scores,
            similarity_scores=all_sim_scores,
            human_scores=all_human_scores,
        )

    def _calculate_metrics(
        self,
        distances: SummarizationDistances,
    ) -> SummarizationMetrics:
        cosine_spearman_scores = []
        cosine_pearson_scores = []
        dot_spearman_scores = []
        dot_pearson_scores = []
        pearson_scores = []
        spearman_scores = []

        for human_scores, cosine_pred_scores, dot_pred_scores, sim_scores in zip(
            distances["human_scores"],
            distances["cosine_scores"],
            distances["dot_scores"],
            distances["similarity_scores"],
            strict=True,
        ):
            cosine_spearman_scores.append(
                spearmanr(human_scores, cosine_pred_scores).statistic
            )
            cosine_pearson_scores.append(
                pearsonr(human_scores, cosine_pred_scores).statistic
            )
            dot_spearman_scores.append(
                spearmanr(human_scores, dot_pred_scores).statistic
            )
            dot_pearson_scores.append(pearsonr(human_scores, dot_pred_scores).statistic)
            spearman_scores.append(spearmanr(human_scores, sim_scores).statistic)
            pearson_scores.append(pearsonr(human_scores, sim_scores).statistic)

        return SummarizationMetrics(
            pearson=float(np.mean(pearson_scores)),
            spearman=float(np.mean(spearman_scores)),
            cosine_spearman=float(np.mean(cosine_spearman_scores)),
            cosine_pearson=float(np.mean(cosine_pearson_scores)),
            dot_pearson=float(np.mean(dot_pearson_scores)),
            dot_spearman=float(np.mean(dot_spearman_scores)),
        )


@deprecated(
    "The used Evaluator is deprecated due to a bug (https://github.com/embeddings-benchmark/mteb/issues/1156). Use the latest version of the dataset to use the latest version of the Evaluator."
)
class DeprecatedSummarizationEvaluator(SummarizationEvaluator):
    """A deprecated version of the SummarizationEvaluator that contains the bug outlines in https://github.com/embeddings-benchmark/mteb/issues/1156.

    It is kept here to maintain compatibility with older versions of the benchmark, but we do not recommend using it.
    """

    def _calculate_metrics(
        self,
        distances: SummarizationDistances,
    ) -> SummarizationMetrics:
        cosine_spearman_scores = []
        cosine_pearson_scores = []
        dot_spearman_scores = []
        dot_pearson_scores = []
        pearson_scores = []
        spearman_scores = []

        for human_scores, cosine_pred_scores, dot_pred_scores, sim_scores in zip(
            distances["human_scores"],
            distances["cosine_scores"],
            distances["dot_scores"],
            distances["similarity_scores"],
            strict=True,
        ):
            cosine_spearman_scores.append(spearmanr(human_scores, cosine_pred_scores))
            cosine_pearson_scores.append(pearsonr(human_scores, cosine_pred_scores))
            dot_spearman_scores.append(spearmanr(human_scores, dot_pred_scores))
            dot_pearson_scores.append(pearsonr(human_scores, dot_pred_scores))
            spearman_scores.append(spearmanr(human_scores, sim_scores))
            pearson_scores.append(pearsonr(human_scores, sim_scores))

        return SummarizationMetrics(
            pearson=float(np.mean(pearson_scores)),
            spearman=float(np.mean(spearman_scores)),
            cosine_spearman=float(np.mean(cosine_spearman_scores)),
            cosine_pearson=float(np.mean(cosine_pearson_scores)),
            dot_pearson=float(np.mean(dot_pearson_scores)),
            dot_spearman=float(np.mean(dot_spearman_scores)),
        )

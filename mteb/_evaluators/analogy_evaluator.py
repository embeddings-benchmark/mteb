from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mteb._create_dataloaders import create_dataloader

from .evaluator import Evaluator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from datasets import Dataset

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models import EncoderProtocol
    from mteb.timing import TimingStack
    from mteb.types import EncodeKwargs

logger = logging.getLogger(__name__)


class AnalogyEvaluator(Evaluator):
    """Given an analogy A:B::C:D, ranks the cosine similarity of D to B - A + C, relative to all other concepts.

    Every input of a concept is embedded, and the embeddings of the inputs of the same concept are mean-pooled
    and L2-normalised. A, B, and C are excluded.
    """

    def __init__(
        self,
        analogies: Dataset,
        concepts: Dataset,
        *,
        analogy_column_names: Sequence[str],
        concept_column_name: str,
        input_column_name: str,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        timer: TimingStack,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.analogies = analogies
        self.concepts = concepts
        self.analogy_column_names = analogy_column_names
        self.concept_column_name = concept_column_name
        self.input_column_name = input_column_name
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset
        self.timer = timer

    def __call__(
        self,
        model: EncoderProtocol,
        *,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None = None,
    ) -> list[int]:
        """Return the rank of the most similar concept, where rank 1 means it is the most similar candidate."""
        modality = self.task_metadata.modalities[0]
        with self.timer(
            "Encoding concepts",
            split=self.hf_split,
            subset=self.hf_subset,
            log_message="Running analogy - Encoding concepts...",
        ):
            inputs = self.concepts.select_columns(self.input_column_name)
            if self.input_column_name != modality:
                inputs = inputs.rename_column(self.input_column_name, modality)
            embeddings = model.encode(
                create_dataloader(
                    inputs,
                    task_metadata=self.task_metadata,
                    num_proc=num_proc,
                    **encode_kwargs,
                ),
                task_metadata=self.task_metadata,
                hf_split=self.hf_split,
                hf_subset=self.hf_subset,
                **encode_kwargs,
            )

        with self.timer(
            "Scoring",
            split=self.hf_split,
            subset=self.hf_subset,
            log_message="Running analogy - Ranking answers...",
        ):
            concept_names = sorted(set(self.concepts[self.concept_column_name]))
            concept_index = {name: i for i, name in enumerate(concept_names)}
            concept_embeddings = _mean_pool(
                np.asarray(embeddings, dtype=np.float32),
                np.array(
                    [
                        concept_index[name]
                        for name in self.concepts[self.concept_column_name]
                    ]
                ),
                len(concept_names),
            )
            a, b, c, d = (
                np.array([concept_index[name] for name in self.analogies[column]])
                for column in self.analogy_column_names
            )
            ranks = _rank_answers(concept_embeddings, a, b, c, d)

        logger.info("Running analogy - Finished.")
        return cast("list[int]", ranks.tolist())


def _mean_pool(embeddings: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """Average the embeddings of each group, then L2-normalise the averages."""
    sums = np.zeros((n_groups, embeddings.shape[1]), dtype=embeddings.dtype)
    np.add.at(sums, groups, embeddings)
    means = sums / np.bincount(groups, minlength=n_groups)[:, None]
    normalized: np.ndarray = means / np.linalg.norm(means, axis=1, keepdims=True)
    return normalized


def _rank_answers(
    concepts: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Rank D among all concepts by cosine similarity to B - A + C, excluding A, B, and C from the distractor pool.

    The rank is 1 + the number of candidates that are strictly more similar than D.
    """
    target = concepts[b] - concepts[a] + concepts[c]
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    similarities = target @ concepts.T
    rows = np.arange(len(d))
    answer_similarities = similarities[rows, d]
    for question in (a, b, c):
        similarities[rows, question] = -np.inf
    ranks: np.ndarray = 1 + (similarities > answer_similarities[:, None]).sum(axis=1)
    return ranks

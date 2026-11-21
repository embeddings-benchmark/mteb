from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ScoringFunction

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from mteb.models import MTEBModels
    from mteb.timing import TimingStack
    from mteb.types import EncodeKwargs, HFSubset, ScoresDict

_SUPPORTED_SIMILARITIES = (ScoringFunction.COSINE, ScoringFunction.DOT_PRODUCT, None)

_ANALOGY_COLUMNS = ("word1", "word2", "word3", "word4")

_VISLINGA_METADATA: dict[str, Any] = {
    "reference": "https://arxiv.org/abs/2610.25252",
    "dataset": {
        "path": "YGideoni/VisLingA",
        "revision": "428b306543b1ab693516a33c49ec825171c4f0da",
        "name": "analogies",
    },
    "type": "Reranking",
    "eval_splits": ["test"],
    "eval_langs": ["eng-Latn"],
    "main_score": "analogy_hit_rate_at_10",
    "date": ("2013-01-16", "2026-10-14"),
    "domains": ["Encyclopaedic"],
    "task_subtypes": [],
    "license": "multiple",
    "annotations_creators": "derived",
    "dialect": [],
    "sample_creation": "found",
    "bibtex_citation": r"""
@article{gideoni2026persistent,
  author = {Gideoni, Yonatan and Gelberg, Yoav and Rudner, Tim G. J. and Gal, Yarin},
  journal = {arXiv preprint arXiv:2610.25252},
  title = {Persistent Representational Misalignment Induces Vision-Language Model Failures},
  year = {2026},
}
""",
}


class _VisLingAReranking(AbsTaskRetrieval):
    """Analogies A:B::C:D over concepts, each represented by one or more inputs (words or images).

    Queries and corpus are both the concept inputs, so the standard retrieval evaluation scores every input against
    every input. `task_specific_scores` rebuilds the input x input similarity matrix from the results, pools it into
    concept cosine similarities and ranks the answer D of each analogy among all concepts except A, B and C by
    M[B, X] - M[A, X] + M[C, X], i.e. by cosine similarity to B - A + C. This requires an inner-product similarity
    (cosine or dot product), so models with other scoring functions (e.g. late interaction) are unsupported for now.
    """

    abstask_prompt = "Represent the concept."
    _support_cross_encoder = False
    _support_search = False

    def _load_concept_inputs(
        self, analogies: Dataset, split: str, num_proc: int | None
    ) -> Dataset:
        """The inputs representing each concept, with a `concept` column and one column named after the modality."""
        raise NotImplementedError

    def _input_ids(self, concept_inputs: Dataset) -> list[str]:
        raise NotImplementedError

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        analogies = load_dataset(
            self.metadata.dataset["path"],
            self.metadata.dataset["name"],
            revision=self.metadata.dataset["revision"],
            num_proc=num_proc,
        )
        self.input_concepts: dict[str, list[str]] = {}
        self.input_ids: dict[str, list[str]] = {}
        self.analogies: dict[str, tuple[list[str], ...]] = {}
        self.dataset = {"default": {}}
        modality = self.metadata.modalities[0]
        for split in self.metadata.eval_splits:
            concept_inputs = self._load_concept_inputs(
                analogies[split], split, num_proc
            )
            concepts = list(concept_inputs["concept"])
            a, b, c, d = (list(analogies[split][column]) for column in _ANALOGY_COLUMNS)
            missing = (set(a) | set(b) | set(c) | set(d)) - set(concepts)
            if missing:
                raise ValueError(
                    f"Analogies use concepts without inputs: {sorted(missing)[:10]}"
                )
            ids = self._input_ids(concept_inputs)
            inputs = concept_inputs.select_columns([modality]).add_column("id", ids)
            by_concept: dict[str, list[str]] = {}
            for input_id, concept in zip(ids, concepts, strict=True):
                by_concept.setdefault(concept, []).append(input_id)
            self.dataset["default"][split] = RetrievalSplitData(
                queries=inputs,
                corpus=inputs,
                relevant_docs={
                    input_id: dict.fromkeys(by_concept[concept], 1)
                    for input_id, concept in zip(ids, concepts, strict=True)
                },
                top_ranked=None,
            )
            self.input_concepts[split] = concepts
            self.input_ids[split] = ids
            self.analogies[split] = (a, b, c, d)
        self.data_loaded = True

    def evaluate(
        self,
        model: MTEBModels,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: EncodeKwargs,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        timer: TimingStack | None = None,
        **kwargs: Any,
    ) -> Mapping[HFSubset, ScoresDict]:
        similarity = model.mteb_model_meta.similarity_fn_name
        if similarity not in _SUPPORTED_SIMILARITIES:
            raise ValueError(
                f"{self.metadata.name} pools concept similarities from pairwise input similarities, which requires a "
                f"cosine or dot-product similarity, but {model.mteb_model_meta.name} uses {similarity}."
            )
        return super().evaluate(
            model,
            split,
            subsets_to_run,
            encode_kwargs=encode_kwargs,
            prediction_folder=prediction_folder,
            num_proc=num_proc,
            timer=timer,
            **kwargs,
        )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        similarities = _similarity_matrix(results, self.input_ids[hf_split])
        concept_similarities, names = _pool_similarities(
            similarities, self.input_concepts[hf_split]
        )
        ranks = _analogy_ranks(concept_similarities, names, self.analogies[hf_split])
        return {
            "analogy_hit_rate_at_1": float(np.mean(ranks == 1)),
            "analogy_hit_rate_at_10": float(np.mean(ranks <= 10)),
            "analogy_avg_rank": float(np.mean(ranks)),
            "analogy_median_rank": float(np.median(ranks)),
        }


def _similarity_matrix(
    results: dict[str, dict[str, float]], ids: list[str]
) -> np.ndarray:
    """Input x input similarities from the retrieval results, which must contain every pair."""
    index = {input_id: i for i, input_id in enumerate(ids)}
    n = len(ids)
    similarities = np.full((n, n), np.nan, dtype=np.float64)
    for qid, doc_scores in results.items():
        cols = np.fromiter(
            (index[doc_id] for doc_id in doc_scores),
            dtype=np.int64,
            count=len(doc_scores),
        )
        similarities[index[qid], cols] = np.fromiter(
            doc_scores.values(), dtype=np.float64, count=len(doc_scores)
        )
    if np.isnan(similarities).any():
        raise ValueError(
            f"{int(np.isnan(similarities).sum())} input pairs are missing from the results"
        )
    return similarities


def _pool_similarities(
    similarities: np.ndarray, concepts: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Cosine similarity of concept embeddings, each the mean of its input embeddings.

    For an inner-product similarity S, M[A, X] = mean_{i in A, j in X} S[i, j] / (|mean A| |mean X|), with
    |mean A|^2 = mean_{i, i' in A} S[i, i']. The mean is over the L2-normalised embeddings for cosine models and over
    the raw embeddings for dot-product models.
    """
    names = sorted(set(concepts))
    index = {name: i for i, name in enumerate(names)}
    groups = np.array([index[concept] for concept in concepts])
    membership = np.zeros((len(names), len(concepts)), dtype=np.float64)
    membership[groups, np.arange(len(concepts))] = 1.0
    membership /= membership.sum(axis=1, keepdims=True)
    pooled = membership @ similarities @ membership.T
    norms = np.sqrt(np.diag(pooled))
    return pooled / np.outer(norms, norms), names


def _analogy_ranks(
    concept_similarities: np.ndarray, names: list[str], analogies: tuple[list[str], ...]
) -> np.ndarray:
    """Rank of D among all concepts except A, B and C by M[B, X] - M[A, X] + M[C, X]."""
    index = {name: i for i, name in enumerate(names)}
    a, b, c, d = (np.array([index[w] for w in column]) for column in analogies)
    scores = concept_similarities[b] - concept_similarities[a] + concept_similarities[c]
    rows = np.arange(len(d))
    for excluded in (a, b, c):
        scores[rows, excluded] = -np.inf
    answer = scores[rows, d][:, None]
    name_array = np.array(names)
    tied_above = (scores == answer) & (name_array[None, :] > name_array[d][:, None])
    ranks: np.ndarray = 1 + (scores > answer).sum(axis=1) + tied_above.sum(axis=1)
    return ranks


class VisLingAText(_VisLingAReranking):
    metadata = TaskMetadata(
        name="VisLingAText",
        description=(
            "Word analogies (A is to B as C is to D) over 746 concepts from the analogy dataset of Mikolov et al. "
            "(2013), e.g. capitals, currencies and grammatical forms. Each concept is represented by the embedding "
            "of its name, and all concepts except A, B and C are reranked by cosine similarity to B - A + C."
        ),
        category="t2t",
        modalities=["text"],
        **_VISLINGA_METADATA,
    )

    def _load_concept_inputs(  # noqa: PLR6301
        self, analogies: Dataset, split: str, num_proc: int | None
    ) -> Dataset:
        names = sorted(
            {word for column in _ANALOGY_COLUMNS for word in analogies[column]}
        )
        return Dataset.from_dict({"concept": names, "text": names})

    def _input_ids(self, concept_inputs: Dataset) -> list[str]:  # noqa: PLR6301
        return list(concept_inputs["concept"])


class VisLingAImage(_VisLingAReranking):
    # top_k = max(k_values) must cover all 7,365 inputs so that the results hold every pair
    k_values = (1, 3, 5, 10, 20, 100, 1000, 10000)

    metadata = TaskMetadata(
        name="VisLingAImage",
        description=(
            "Visual analogies (A is to B as C is to D) over 746 concepts from the analogy dataset of Mikolov et al. "
            "(2013), e.g. capitals, currencies and grammatical forms. Each concept is represented by the mean of the "
            "L2-normalised embeddings of several Wikimedia Commons images of that concept, and all concepts except A, "
            "B and C are reranked by cosine similarity to B - A + C."
        ),
        category="i2i",
        modalities=["image"],
        **_VISLINGA_METADATA,
    )

    def _load_concept_inputs(
        self, analogies: Dataset, split: str, num_proc: int | None
    ) -> Dataset:
        return load_dataset(
            self.metadata.dataset["path"],
            "images",
            revision=self.metadata.dataset["revision"],
            split=split,
            num_proc=num_proc,
        ).select_columns(["concept", "image"])

    def _input_ids(self, concept_inputs: Dataset) -> list[str]:  # noqa: PLR6301
        return [f"img{i}" for i in range(len(concept_inputs))]

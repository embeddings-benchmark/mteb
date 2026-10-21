from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from datasets import Dataset, load_dataset

from mteb._create_dataloaders import create_dataloader
from mteb.abstasks._statistics_calculation import (
    calculate_relevant_docs_statistics,
    calculate_single_input_modality_statistics,
    calculate_top_ranked_statistics,
)
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models import EncoderProtocol
from mteb.types import PromptType
from mteb.types.statistics import RetrievalDescriptiveStatistics

if TYPE_CHECKING:
    from pathlib import Path

    import torch
    from torch.utils.data import DataLoader

    from mteb.models import MTEBModels
    from mteb.models.model_meta import ModelMeta
    from mteb.timing import TimingStack
    from mteb.types import Array, BatchedInput, EncodeKwargs, ScoresDict

logger = logging.getLogger(__name__)

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
    "main_score": "hit_rate_at_10",
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
    """Analogies A:B::C:D as reranking: each analogy is a query, the concepts are the corpus and D is the relevant
    document. The candidates of a query are all concepts except A, B and C.

    A concept is represented by one or more inputs, whose embeddings are mean-pooled and L2-normalised. The query
    embedding is B - A + C (L2-normalised) and candidates are scored by cosine similarity.
    """

    abstask_prompt = "Represent the concept."

    def _load_concept_inputs(
        self, analogies: Dataset, split: str, num_proc: int | None
    ) -> Dataset:
        """The inputs representing each concept, with a `concept` column and one column named after the modality."""
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
        self.concept_inputs: dict[str, Dataset] = {}
        self.query_terms: dict[str, dict[str, tuple[str, str, str]]] = {}
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            concept_inputs = self._load_concept_inputs(
                analogies[split], split, num_proc
            )
            self.concept_inputs[split] = concept_inputs
            names = sorted(set(concept_inputs["concept"]))

            a, b, c, d = (analogies[split][column] for column in _ANALOGY_COLUMNS)
            missing = (set(a) | set(b) | set(c) | set(d)) - set(names)
            if missing:
                raise ValueError(
                    f"Analogies use concepts without inputs: {sorted(missing)[:10]}"
                )
            query_ids = [f"q{i}" for i in range(len(a))]
            self.dataset["default"][split] = RetrievalSplitData(
                queries=Dataset.from_dict(
                    {
                        "id": query_ids,
                        "text": [
                            f"{x}:{y}::{z}" for x, y, z in zip(a, b, c, strict=True)
                        ],
                    }
                ),
                corpus=Dataset.from_dict({"id": names, "text": names}),
                relevant_docs={
                    qid: {answer: 1} for qid, answer in zip(query_ids, d, strict=True)
                },
                top_ranked={
                    qid: _candidates(names, excluded={x, y, z})
                    for qid, x, y, z in zip(query_ids, a, b, c, strict=True)
                },
            )
            self.query_terms[split] = {
                qid: (x, y, z) for qid, x, y, z in zip(query_ids, a, b, c, strict=True)
            }
        self.data_loaded = True

    def _evaluate_subset(  # type: ignore[override]
        self,
        model: MTEBModels,
        data_split: RetrievalSplitData,
        *,
        encode_kwargs: EncodeKwargs,
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        timer: TimingStack,
        **kwargs: Any,
    ) -> ScoresDict:
        if not isinstance(model, EncoderProtocol):
            raise TypeError("Expected model to be an instance of EncoderProtocol")

        concept_inputs = self.concept_inputs[hf_split]
        with timer(
            "Encoding concepts",
            split=hf_split,
            subset=hf_subset,
            log_message="Running VisLingA - Encoding concepts...",
        ):
            embeddings = model.encode(
                create_dataloader(
                    concept_inputs.select_columns(self.metadata.modalities[0]),
                    task_metadata=self.metadata,
                    num_proc=num_proc,
                    **encode_kwargs,
                ),
                task_metadata=self.metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                **encode_kwargs,
            )
        concept_index = {name: i for i, name in enumerate(data_split["corpus"]["id"])}
        concepts = _mean_pool(
            np.asarray(embeddings, dtype=np.float32),
            np.array([concept_index[name] for name in concept_inputs["concept"]]),
            len(concept_index),
        )
        return super()._evaluate_subset(
            _AnalogyEncoder(model, concepts, concept_index, self.query_terms[hf_split]),
            data_split,
            encode_kwargs=encode_kwargs,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prediction_folder=prediction_folder,
            num_proc=num_proc,
            timer=timer,
            **kwargs,
        )

    def task_specific_scores(  # noqa: PLR6301
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        """Mean and median rank of each query's answer, with exact ties broken by descending document id."""
        ranks = [
            _rank(results[qid], answer)
            for qid, docs in qrels.items()
            for answer in docs
        ]
        return {
            "avg_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
        }


class _AnalogyEncoder:
    """Exposes an encoder's concept embeddings to the standard retrieval pipeline.

    Calculate documents' (concepts) mean-pooled concept embedding, query (analogies A:B::C:?) B - A + C, and
    L2-normalise so we use cosine similarity.
    """

    def __init__(
        self,
        model: EncoderProtocol,
        concepts: np.ndarray,
        concept_index: dict[str, int],
        query_terms: dict[str, tuple[str, str, str]],
    ) -> None:
        self.model = model
        self.concepts = concepts
        self.concept_index = concept_index
        self.query_terms = query_terms

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return self.model.mteb_model_meta

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        ids = [input_id for batch in inputs for input_id in batch["id"]]
        if prompt_type != PromptType.query:
            return self.concepts[[self.concept_index[input_id] for input_id in ids]]
        a, b, c = (
            self.concepts[
                [self.concept_index[self.query_terms[input_id][i]] for input_id in ids]
            ]
            for i in range(3)
        )
        targets = b - a + c
        normalized: np.ndarray = targets / np.linalg.norm(
            targets, axis=1, keepdims=True
        )
        return normalized

    def similarity(self, embeddings1: Array, embeddings2: Array) -> torch.Tensor:  # noqa: PLR6301
        import torch

        return torch.as_tensor(embeddings1) @ torch.as_tensor(embeddings2).T

    def similarity_pairwise(  # noqa: PLR6301
        self, embeddings1: Array, embeddings2: Array
    ) -> torch.Tensor:
        import torch

        return (torch.as_tensor(embeddings1) * torch.as_tensor(embeddings2)).sum(dim=1)


def _candidates(names: list[str], excluded: set[str]) -> list[str]:
    return [name for name in names if name not in excluded]


def _rank(doc_scores: dict[str, float], answer: str) -> int:
    """1 + the number of candidates ranked above the answer, breaking exact ties by descending document id."""
    answer_score = doc_scores[answer]
    ids = np.array(list(doc_scores))
    values = np.fromiter(doc_scores.values(), dtype=float, count=len(doc_scores))
    above = (values > answer_score) | ((values == answer_score) & (ids > answer))
    return 1 + int(above.sum())


def _mean_pool(embeddings: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """Average the embeddings of each group, then L2-normalise the averages."""
    sums = np.zeros((n_groups, embeddings.shape[1]), dtype=embeddings.dtype)
    np.add.at(sums, groups, embeddings)
    means = sums / np.bincount(groups, minlength=n_groups)[:, None]
    normalized: np.ndarray = means / np.linalg.norm(means, axis=1, keepdims=True)
    return normalized


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


class VisLingAImage(_VisLingAReranking):
    metadata = TaskMetadata(
        name="VisLingAImage",
        description=(
            "Visual analogies (A is to B as C is to D) over 746 concepts from the analogy dataset of Mikolov et al. "
            "(2013), e.g. capitals, currencies and grammatical forms. Each concept is represented by the mean "
            "embedding of several Wikimedia Commons images of that concept, and all concepts except A, B and C are "
            "reranked by cosine similarity to B - A + C."
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

    def _calculate_descriptive_statistics_from_split(
        self,
        split: str,
        *,
        hf_subset: str | None = None,
        compute_overall: bool = False,
        num_proc: int | None = None,
    ) -> RetrievalDescriptiveStatistics:
        # The documents are concepts whose embedding is the mean over several images, so describe the images.
        split_data = self.dataset["default"][split]
        queries, corpus = split_data["queries"], split_data["corpus"]
        image_statistics = calculate_single_input_modality_statistics(
            {"image": self.concept_inputs[split]["image"]}, max_workers=num_proc
        )["image_statistics"]
        return RetrievalDescriptiveStatistics(
            num_samples=len(corpus) + len(queries),
            num_queries=len(queries),
            num_documents=len(corpus),
            number_of_characters=0,
            documents_text_statistics=None,
            documents_image_statistics=image_statistics,
            documents_audio_statistics=None,
            documents_video_statistics=None,
            queries_text_statistics=None,
            queries_image_statistics=None,
            queries_audio_statistics=None,
            queries_video_statistics=None,
            relevant_docs_statistics=calculate_relevant_docs_statistics(
                split_data["relevant_docs"], set(queries["id"]), set(corpus["id"])
            ),
            top_ranked_statistics=calculate_top_ranked_statistics(
                split_data["top_ranked"], len(queries)
            ),
        )

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from datasets import Dataset, DatasetDict
from scipy.integrate import trapezoid
from sklearn.metrics.pairwise import euclidean_distances

from mteb._create_dataloaders import create_dataloader
from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models import EncoderProtocol

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from numpy.typing import NDArray

    from mteb.models import MTEBModels
    from mteb.timing import TimingStack
    from mteb.types import EncodeKwargs


class StakeholderGroundingMetrics(TypedDict):
    """Metrics for StakeholderGrounding

    Attributes:
        alpha_auc: Area under the triplet-based Krippendorff's alpha curve, integrated over the log of the distance-ratio
            threshold and normalized to [-1, 1].
    """

    alpha_auc: float


def _triplet_indices(n: int) -> tuple[NDArray[np.int64], ...]:
    """All (anchor, i, j) triplets with i < j and anchor not in {i, j}."""
    i_idx, j_idx = np.triu_indices(n, k=1)
    anchors = np.arange(n)[None, :].repeat(len(i_idx), axis=0)
    valid = (anchors != i_idx[:, None]) & (anchors != j_idx[:, None])
    pair_idx, anchor = np.nonzero(valid)
    return anchor, i_idx[pair_idx], j_idx[pair_idx]


def _arrangement_triplets(
    distances: NDArray[np.floating], similarities: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Distance ratio and model agreement for every triplet in a single arrangement.

    Returns:
        ratio: distance from the anchor to the farther item divided by the distance to the closer item (human layout).
        agree: whether the model is strictly more similar to the item the human placed closer to the anchor.
    """
    anchor, i, j = _triplet_indices(len(distances))
    dist_i, dist_j = distances[anchor, i], distances[anchor, j]
    i_closer = dist_i < dist_j
    closer = np.where(i_closer, i, j)
    farther = np.where(i_closer, j, i)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.maximum(dist_i, dist_j) / np.minimum(dist_i, dist_j)
    agree = similarities[anchor, closer] > similarities[anchor, farther]
    return ratio, agree


def _alpha_auc(
    ratio: NDArray[np.floating],
    agree: NDArray[np.bool_],
    thresholds: NDArray[np.floating],
) -> float:
    """Normalized area under alpha(d) over log(d), where alpha(d) = 2 * agreement - 1 on triplets with ratio > d."""
    kept_thresholds, alphas = [], []
    for threshold in thresholds:
        mask = ratio > threshold
        if mask.any():
            kept_thresholds.append(threshold)
            alphas.append(2 * agree[mask].mean() - 1)
    log_thresholds = np.log(kept_thresholds)
    span = log_thresholds[-1] - log_thresholds[0]
    if span == 0:
        return float(np.mean(alphas))
    return float(trapezoid(alphas, log_thresholds) / span)


def _arrangements_to_pairs(arrangements: Dataset) -> Dataset:
    """One row per pair of texts within an arrangement, scored by 1 - distance / max distance on the canvas."""
    pairs: dict[str, list[Any]] = {
        "round_id": [],
        "sentence1": [],
        "sentence2": [],
        "score": [],
    }
    for round_id, texts, x, y in zip(
        arrangements["round_id"],
        arrangements["texts"],
        arrangements["x"],
        arrangements["y"],
        strict=True,
    ):
        if len(set(texts)) != len(texts):
            raise ValueError(f"Arrangement {round_id} contains duplicate texts")
        distances = euclidean_distances(np.column_stack([x, y]).astype(float))
        i, j = np.triu_indices(len(texts), k=1)
        pairs["round_id"].extend([round_id] * len(i))
        pairs["sentence1"].extend(texts[k] for k in i)
        pairs["sentence2"].extend(texts[k] for k in j)
        pairs["score"].extend((1 - distances[i, j] / distances.max()).tolist())
    return Dataset.from_dict(pairs)


def _pairs_to_arrangements(
    round_ids: Iterable[str],
    text1_idx: Iterable[int],
    text2_idx: Iterable[int],
    scores: Iterable[float],
) -> list[tuple[list[int], NDArray[np.floating]]]:
    """Group the pairs back into arrangements: the text indices of each arrangement and their distance matrix."""
    arrangements: dict[str, list[tuple[int, int, float]]] = {}
    for round_id, idx1, idx2, score in zip(
        round_ids, text1_idx, text2_idx, scores, strict=True
    ):
        arrangements.setdefault(round_id, []).append((idx1, idx2, score))

    result = []
    for pairs in arrangements.values():
        idx = list(dict.fromkeys(i for pair in pairs for i in pair[:2]))
        position = {text_idx: pos for pos, text_idx in enumerate(idx)}
        distances = np.zeros((len(idx), len(idx)))
        for idx1, idx2, score in pairs:
            pos1, pos2 = position[idx1], position[idx2]
            distances[pos1, pos2] = distances[pos2, pos1] = 1 - score
        result.append((idx, distances))
    return result


class StakeholderGrounding(AbsTaskSTS):
    metadata = TaskMetadata(
        name="StakeholderGrounding",
        description="Danish technology practitioners' statements on challenges with Responsible AI, spatially arranged by domain experts (Spatial Arrangement Method) so that canvas proximity reflects perceived similarity. Each sample is a pair of statements from one expert arrangement of 20 statements, scored by their normalized proximity on the canvas. Models are scored by triplet-based Krippendorff's alpha AUC against each expert layout.",
        reference="https://arxiv.org/abs/2605.27168",
        dataset={
            "path": "ryzzlestrizzle/stakeholder-grounding",
            "revision": "836f23984a2761a3a518c6c2582736779935f457",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={"rai": ["dan-Latn"]},
        main_score="alpha_auc",
        date=("2025-06-03", "2025-06-03"),  # TODO: survey and workshop collection dates
        domains=["Government", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@misc{rystrom2026grounding,
  archiveprefix = {arXiv},
  author = {Jonathan Rystrøm and Sofie Burgos-Thorsen and Zihao Fu and Johan Irving Søltoft and Kenneth C. Enevoldsen and Chris Russell},
  eprint = {2605.27168},
  primaryclass = {cs.CL},
  title = {Grounding Text Embeddings in Stakeholder Associations},
  year = {2026},
}
""",
    )

    min_score = 0
    max_score = 1
    thresholds: NDArray[np.floating] = np.logspace(0, np.log10(6.5), 30)

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        self.dataset = {
            hf_subset: DatasetDict(
                {
                    split: _arrangements_to_pairs(arrangements)
                    for split, arrangements in splits.items()
                }
            )
            for hf_subset, splits in self.dataset.items()
        }

    def _evaluate_subset(  # type: ignore[override]
        self,
        model: MTEBModels,
        data_split: Dataset,
        *,
        encode_kwargs: EncodeKwargs,
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        timer: TimingStack,
        **kwargs: Any,
    ) -> StakeholderGroundingMetrics:
        if not isinstance(model, EncoderProtocol):
            raise TypeError("Expected model to be an instance of EncoderProtocol")

        sentences1, sentences2 = (
            list(data_split["sentence1"]),
            list(data_split["sentence2"]),
        )
        unique_texts = list(dict.fromkeys(sentences1 + sentences2))
        text_to_idx = {text: idx for idx, text in enumerate(unique_texts)}

        with timer(
            "Encoding",
            split=hf_split,
            subset=hf_subset,
            log_message="Running stakeholder grounding - Encoding samples...",
        ):
            embeddings = model.encode(
                create_dataloader(
                    Dataset.from_dict({"text": unique_texts}),
                    task_metadata=self.metadata,
                    input_column="text",
                    num_proc=num_proc,
                    **encode_kwargs,
                ),
                task_metadata=self.metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                **encode_kwargs,
            )

        with timer(
            "Scoring",
            split=hf_split,
            subset=hf_subset,
            log_message="Running stakeholder grounding - Computing triplet agreement...",
        ):
            ratios, agreements = [], []
            for idx, distances in _pairs_to_arrangements(
                data_split["round_id"],
                [text_to_idx[text] for text in sentences1],
                [text_to_idx[text] for text in sentences2],
                data_split["score"],
            ):
                similarities = np.asarray(
                    model.similarity(embeddings[idx], embeddings[idx]), dtype=float
                )
                ratio, agree = _arrangement_triplets(distances, similarities)
                ratios.append(ratio)
                agreements.append(agree)

            alpha_auc = _alpha_auc(
                np.concatenate(ratios), np.concatenate(agreements), self.thresholds
            )
        return StakeholderGroundingMetrics(alpha_auc=alpha_auc)

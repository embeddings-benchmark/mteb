from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from mteb.types import RelevantDocumentsType

_HEADING_RE = re.compile(r"(?m)^#+\s.*$")


def _source_grade_to_gain(grade: int) -> int:
    """Convert RAVENEA's [-3, 3] grade to the gain used by its scorer."""
    if not -3 <= grade <= 3:
        raise ValueError(f"RAVENEA grade must be in [-3, 3], got {grade}")
    return 2 ** (grade + 3) - 1


def _source_gain_to_grade(gain: int) -> int:
    shifted = math.log2(gain + 1)
    if not shifted.is_integer():
        raise ValueError(f"Invalid RAVENEA gain: {gain}")
    return int(shifted) - 3


def _dcg(gains: list[int]) -> float:
    return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))


def _source_metrics(
    qrels: RelevantDocumentsType,
    results: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Reproduce the metrics in the official RAVENEA ``gdeval.py``.

    RAVENEA precision and MRR count only documents with the maximum grade for
    each query. Its nDCG first shifts grades by three and then applies an
    exponential gain. The normalized Hub qrels store that gain directly, so
    standard MTEB nDCG is equivalent while the source P@k and MRR are exposed
    here as task-specific scores.
    """
    precisions = {1: [], 3: [], 5: []}
    ndcgs = {1: [], 3: [], 5: []}
    reciprocal_ranks: list[float] = []

    for query_id, doc_scores in results.items():
        if query_id not in qrels:
            continue
        ranked_ids = [
            doc_id
            for doc_id, _ in sorted(
                doc_scores.items(), key=lambda item: item[1], reverse=True
            )
        ]
        query_qrels = qrels[query_id]
        max_gain = max(query_qrels.values(), default=0)
        max_grade = _source_gain_to_grade(max_gain)
        maximally_relevant = {
            doc_id for doc_id, gain in query_qrels.items() if gain == max_gain
        }

        if max_grade > 0:
            reciprocal_rank = 0.0
            for rank, doc_id in enumerate(ranked_ids, start=1):
                if doc_id in maximally_relevant:
                    reciprocal_rank = 1.0 / rank
                    break
            reciprocal_ranks.append(reciprocal_rank)

        # The source scorer assigns grade zero to an unlisted prediction.
        ranked_gains = [
            query_qrels.get(doc_id, _source_grade_to_gain(0)) for doc_id in ranked_ids
        ]
        for k, precision_values in precisions.items():
            top_ids = ranked_ids[:k]
            precision_values.append(
                len(set(top_ids) & maximally_relevant) / len(top_ids)
                if top_ids
                else 0.0
            )
            predicted_gains = ranked_gains[:k]
            ideal_gains = sorted(ranked_gains, reverse=True)[:k]
            ideal_dcg = _dcg(ideal_gains)
            ndcgs[k].append(_dcg(predicted_gains) / ideal_dcg if ideal_dcg else 0.0)

    return {
        "ravenea_mrr": (
            sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        ),
        **{
            f"ravenea_precision_at_{k}": sum(values) / len(values)
            for k, values in precisions.items()
        },
        **{
            f"ravenea_ndcg_at_{k}": sum(values) / len(values)
            for k, values in ndcgs.items()
        },
    }


class RAVENEAI2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RAVENEAI2TRetrieval",
        description=(
            "RAVENEA evaluates culturally grounded image-to-text retrieval. "
            "Each test image is used to rerank ten Wikipedia articles selected "
            "by BM25 and judged by human annotators for cultural relevance on "
            "a seven-point scale from -3 to 3. The task uses the paper's "
            "official test split and candidate sets."
        ),
        reference="https://arxiv.org/abs/2505.14462",
        dataset={
            "path": "Cerru02/RAVENEA",
            "revision": "94b1c2ab28a26c8b6af8995d8445e285cc5275ce",
        },
        type="Reranking",
        category="i2t",
        modalities=["image", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ravenea_ndcg_at_5",
        date=("2026-02-12", "2026-02-14"),
        domains=["Encyclopaedic", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="multiple",
        adapted_from=["CVQA", "CCUB"],
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{li2026ravenea,
  author = {Jiaang Li and Yifei Yuan and Wenyan Li and Mohammad Aliannejadi and Daniel Hershcovich and Anders S{\o}gaard and Ivan Vuli{\'c} and Wenxuan Zhang and Paul Pu Liang and Yang Deng and Serge Belongie},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  title = {{RAVENEA}: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding},
  url = {https://openreview.net/forum?id=4zAbkxQ23i},
  year = {2026},
}
""",
        prompt={
            "query": "Retrieve Wikipedia articles that provide culturally relevant context for this image."
        },
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        split_data = self.dataset["default"]["test"]
        query_metadata = [
            column
            for column in ("country", "category", "task_type")
            if column in split_data["queries"].column_names
        ]
        if query_metadata:
            split_data["queries"] = split_data["queries"].remove_columns(query_metadata)

        # Match the official inference code: remove Markdown heading lines and
        # encode the complete remaining article, without captions or labels.
        split_data["corpus"] = split_data["corpus"].map(
            lambda row: {"text": _HEADING_RE.sub("", row["text"]).strip()},
            num_proc=num_proc,
            desc="Normalizing RAVENEA Wikipedia articles",
        )

    def task_specific_scores(  # noqa: PLR6301
        self,
        scores: dict[str, dict[str, float]],
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        return _source_metrics(qrels, results)

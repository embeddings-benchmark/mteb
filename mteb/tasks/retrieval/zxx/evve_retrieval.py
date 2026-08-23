from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


def trapezoidal_average_precision(
    ranking: Sequence[str], positives: set[str], ignored: set[str]
) -> float:
    """Reproduce EVVE's AP integration while removing event-specific nulls."""
    if not positives:
        return 0.0

    positive_ranks: list[int] = []
    ignored_seen = 0
    for rank, corpus_id in enumerate(ranking):
        if corpus_id in positives:
            positive_ranks.append(rank - ignored_seen)
        elif corpus_id in ignored:
            ignored_seen += 1

    recall_step = 1.0 / len(positives)
    average_precision = 0.0
    for positives_seen, rank in enumerate(positive_ranks):
        precision_left = 1.0 if rank == 0 else positives_seen / rank
        precision_right = (positives_seen + 1) / (rank + 1)
        average_precision += (precision_left + precision_right) * recall_step / 2.0
    return average_precision


def evve_scores(
    qrels: Mapping[str, Mapping[str, int]],
    results: Mapping[str, Mapping[str, float]],
    query_events: Mapping[str, str],
    query_ignored: Mapping[str, Iterable[str]],
) -> dict[str, float]:
    """Compute original query-macro and event-balanced EVVE scores."""
    event_average_precisions: dict[str, list[float]] = defaultdict(list)
    all_average_precisions: list[float] = []

    for query_id, relevant in qrels.items():
        ranking = [
            corpus_id
            for corpus_id, _ in sorted(
                results[query_id].items(), key=lambda item: (-item[1], item[0])
            )
        ]
        positives = {corpus_id for corpus_id, score in relevant.items() if score > 0}
        average_precision = trapezoidal_average_precision(
            ranking, positives, set(query_ignored[query_id])
        )
        all_average_precisions.append(average_precision)
        event_average_precisions[query_events[query_id]].append(average_precision)

    event_maps = [
        sum(average_precisions) / len(average_precisions)
        for average_precisions in event_average_precisions.values()
    ]
    return {
        "evve_avg_map": sum(event_maps) / len(event_maps),
        "evve_overall_map": sum(all_average_precisions) / len(all_average_precisions),
    }


class EVVERetrieval(AbsTaskRetrieval):
    # The final value requests the complete frozen corpus for EVVE's source metric.
    k_values = (1, 3, 5, 10, 20, 100, 1000, 1644)

    metadata = TaskMetadata(
        name="EVVERetrieval",
        description=(
            "Video-to-video retrieval of the same specific real-world event. "
            "This fixed surviving-public-media subset of EVVE contains 466 "
            "queries, 1,644 database videos, all 13 events, and 86,925 positive "
            "relevance judgments. It is not the complete original benchmark, "
            "does not include the original 100,000-video distractor collection, "
            "and its scores are not directly comparable with published results."
        ),
        reference="https://openaccess.thecvf.com/content_cvpr_2013/html/Revaud_Event_Retrieval_in_2013_CVPR_paper.html",
        dataset={
            "path": "Cerru02/EVVE",
            "revision": "0a7d35a4358992882a9989a4e64d1f1ce7a7c78c",
        },
        type="Any2AnyRetrieval",
        category="v2v",
        modalities=["video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="evve_avg_map",
        date=("2012-01-01", "2013-06-30"),
        domains=["Web"],
        task_subtypes=["Event Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{revaud2013event,
  author = {Revaud, Jerome and Douze, Matthijs and Schmid, Cordelia and Jegou, Herve},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages = {2459--2466},
  title = {Event Retrieval in Large Video Collections with Circulant Temporal Encoding},
  year = {2013},
}
""",
        prompt={
            "query": "Retrieve other videos depicting the same specific real-world event."
        },
        is_beta=True,
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: Mapping[str, Mapping[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        del scores
        queries = self.dataset[hf_subset][hf_split]["queries"]
        query_events = dict(zip(queries["id"], queries["event"], strict=True))
        query_ignored = dict(
            zip(queries["id"], queries["ignored_corpus_ids"], strict=True)
        )
        return evve_scores(qrels, results, query_events, query_ignored)

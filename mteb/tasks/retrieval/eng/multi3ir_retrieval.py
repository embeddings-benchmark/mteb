from __future__ import annotations

from collections import defaultdict
from typing import Any

import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_K_VALUES = (5, 10, 20, 100)


class Multi3IRRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Multi3IRRetrieval",
        description=(
            "Multi-perspective, multi-domain, multi-modal retrieval. Each query is an open-ended Stack Exchange "
            "question with several perspectives, and the corpus mixes C4 web passages and web images (1.01M "
            "documents). Gold documents are judged per perspective. The main score is hard coverage HC@10: the "
            "fraction of a query's perspectives with at least one gold document in the top 10."
        ),
        reference="https://arxiv.org/abs/2608.30949",
        dataset={
            "path": "seokwon99/Multi3IR-mteb",
            "revision": "206374b4fbf9b192bb5735b6b1d7021deabec70f",
        },
        type="Any2AnyRetrieval",
        category="t2it",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hc_at_10",
        date=("2008-08-01", "2025-12-31"),
        domains=["Web", "Social", "Academic", "Programming"],
        task_subtypes=["Question answering", "Cross-Modal Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{song2026multi3ir,
  author = {Song, Seokwon and Kim, Sohyeon and Kim, Gunhee},
  journal = {arXiv preprint arXiv:2608.30949},
  title = {MULTI3IR: A Benchmark for Multi-perspective Multi-domain Multi-modal Information Retrieval},
  year = {2026},
}
""",
        prompt={"query": "Retrieve images or text relevant to the user's query."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        super().load_data(num_proc=num_proc, **kwargs)
        # gold documents grouped by perspective, for hard coverage
        self.perspectives = {}
        for split in self.metadata.eval_splits:
            rows = datasets.load_dataset(
                self.metadata.dataset["path"],
                "perspectives",
                split=split,
                revision=self.metadata.dataset["revision"],
            )
            groups: dict[str, dict[str, set[str]]] = defaultdict(
                lambda: defaultdict(set)
            )
            for q, d, p in zip(
                rows["query-id"], rows["corpus-id"], rows["perspective"], strict=True
            ):
                groups[q][p].add(d)
            self.perspectives[split] = groups

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        """Hard coverage HC@k: mean over queries of the share of perspectives covered in the top k."""
        groups = self.perspectives[hf_split]
        hc = dict.fromkeys(_K_VALUES, 0.0)
        for query_id in qrels:
            ranked = sorted(
                results.get(query_id, {}).items(), key=lambda x: (-x[1], x[0])
            )
            persps = groups[query_id].values()
            for k in _K_VALUES:
                top = {doc_id for doc_id, _ in ranked[:k]}
                hc[k] += sum(1 for docs in persps if docs & top) / len(persps)
        return {f"hc_at_{k}": hc[k] / len(qrels) for k in _K_VALUES}

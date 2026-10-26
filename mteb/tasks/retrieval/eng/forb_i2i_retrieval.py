from __future__ import annotations

from collections import defaultdict
from typing import Any

from datasets import Image

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FORBI2I(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FORBI2IRetrieval",
        description="Retrieve flat object images from 8 classes.",
        reference="https://github.com/pxiangwu/FORB",
        dataset={
            "path": "mteb/forb_retrieval",
            "revision": "26ab4bd972854becada339afc80f5f3ffc047e2b",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_1",
        date=("2022-01-01", "2023-01-01"),
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        superseded_by="FORBI2IRetrieval.v2",
        bibtex_citation=r"""
@misc{wu2023forbflatobjectretrieval,
  archiveprefix = {arXiv},
  author = {Pengxiang Wu and Siman Wang and Kevin Dela Rosa and Derek Hao Hu},
  eprint = {2309.16249},
  primaryclass = {cs.CV},
  title = {FORB: A Flat Object Retrieval Benchmark for Universal Image Embedding},
  url = {https://arxiv.org/abs/2309.16249},
  year = {2023},
}
""",
    )


class FORBI2IV2(FORBI2I):
    metadata = FORBI2I.metadata.model_copy(
        update={
            "name": "FORBI2IRetrieval.v2",
            "description": FORBI2I.metadata.description
            + " Version 2 drops queries whose image is identical to another query's image with a different answer: 3,074 queries share one 80x80 placeholder image (a failed download), so no model can answer them.",
            "adapted_from": ["FORBI2IRetrieval"],
            "superseded_by": None,
        }
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        for subset in self.dataset:
            for split in self.dataset[subset]:
                data = self.dataset[subset][split]
                queries, relevant_docs = data["queries"], data["relevant_docs"]
                raw_images = queries.cast_column("image", Image(decode=False))["image"]
                images = [image["bytes"] for image in raw_images]
                answers: dict[bytes, set[frozenset[str]]] = defaultdict(set)
                for query_id, image in zip(queries["id"], images, strict=True):
                    answers[image].add(frozenset(relevant_docs[query_id]))
                keep = [i for i, image in enumerate(images) if len(answers[image]) == 1]
                data["queries"] = queries.select(keep)
                kept_ids = set(data["queries"]["id"])
                data["relevant_docs"] = {
                    query_id: docs
                    for query_id, docs in relevant_docs.items()
                    if query_id in kept_ids
                }

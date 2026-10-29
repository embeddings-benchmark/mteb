from __future__ import annotations

import io
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


# Failed downloads were saved as placeholder images (an 80x80 "no image" icon, a 120x90
# thumbnail, a 1x1 pixel, an "image deleted" banner). All of them fit in 130x130 px,
# and every real image in FORB has a longer side.
_PLACEHOLDER_MAX_SIDE = 130


def _is_placeholder(image: dict[str, Any]) -> bool:
    import PIL.Image

    return max(PIL.Image.open(io.BytesIO(image["bytes"])).size) <= _PLACEHOLDER_MAX_SIDE


class FORBI2IV2(AbsTaskRetrieval):
    metadata = FORBI2I.metadata.model_copy(
        update={
            "name": "FORBI2IRetrieval.v2",
            "description": FORBI2I.metadata.description
            + " Version 2 drops placeholder images left by failed downloads (an 80x80 'no image' icon used by 3,074 queries, a 1x1 pixel, an 'image deleted' banner) from queries and corpus, and the queries left without a correct answer.",
            "adapted_from": ["FORBI2IRetrieval"],
            "superseded_by": None,
        }
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        for subset in self.dataset.values():
            for data in subset.values():
                for side in ("queries", "corpus"):
                    data[side] = (
                        data[side]
                        .cast_column("image", Image(decode=False))
                        .filter(
                            lambda image: not _is_placeholder(image),
                            input_columns="image",
                            num_proc=num_proc,
                        )
                        .cast_column("image", Image())
                    )
                corpus_ids = set(data["corpus"]["id"])
                relevant_docs = {}
                for query_id, docs in data["relevant_docs"].items():
                    kept = {
                        doc: score for doc, score in docs.items() if doc in corpus_ids
                    }
                    if kept:
                        relevant_docs[query_id] = kept
                query_ids = data["queries"]["id"]
                data["queries"] = data["queries"].select(
                    [
                        i
                        for i, query_id in enumerate(query_ids)
                        if query_id in relevant_docs
                    ]
                )
                data["relevant_docs"] = {
                    query_id: relevant_docs[query_id]
                    for query_id in data["queries"]["id"]
                }

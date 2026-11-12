from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SketchyI2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SketchyI2IRetrieval",
        description="Retrieve photos from sketches.",
        reference="https://arxiv.org/abs/2202.01747",
        dataset={
            "path": "mteb/sketchy",
            "revision": "f9456d4ef865495dacd379c7f43925f4155831bf",
        },
        type="Any2AnyRetrieval",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_1",
        date=("2021-12-06", "2021-12-14"),  # conference dates
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        superseded_by="SketchyI2IRetrieval.v2",
        bibtex_citation=r"""
@inproceedings{ypsilantis2021met,
  author = {Ypsilantis, Nikolaos-Antonios and Garcia, Noa and Han, Guangxing and Ibrahimi, Sarah and Van Noord, Nanne and Tolias, Giorgos},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  title = {The met dataset: Instance-level recognition for artworks},
  year = {2021},
}
""",
    )
    skip_first_result = False


class SketchyI2IRetrievalV2(AbsTaskRetrieval):
    metadata = SketchyI2IRetrieval.metadata.model_copy(
        update={
            "name": "SketchyI2IRetrieval.v2",
            "description": "Retrieve photos from sketches. Version 2 keeps one image per id: v1 has 6 augmented copies of each sketch and 2 of each photo under the same id, so models encoded each item several times.",
            "adapted_from": ["SketchyI2IRetrieval"],
            "superseded_by": None,
        }
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        for subset in self.dataset.values():
            for split in subset.values():
                for side in ("queries", "corpus"):
                    first_rows: dict[str, int] = {}
                    for i, row_id in enumerate(split[side]["id"]):
                        first_rows.setdefault(row_id, i)
                    split[side] = split[side].select(list(first_rows.values()))

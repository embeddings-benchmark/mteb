from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class METI2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="METI2IRetrieval",
        description="Retrieve photos of more than 224k artworks.",
        reference="https://arxiv.org/abs/2202.01747",
        dataset={
            "path": "mteb/met",
            "revision": "c4994f2da3df0a6e20a310048a04be49da7282c3",
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
        bibtex_citation=r"""
@inproceedings{ypsilantis2021met,
  author = {Ypsilantis, Nikolaos-Antonios and Garcia, Noa and Han, Guangxing and Ibrahimi, Sarah and Van Noord, Nanne and Tolias, Giorgos},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  title = {The met dataset: Instance-level recognition for artworks},
  year = {2021},
}
""",
    )
    skip_first_result = True

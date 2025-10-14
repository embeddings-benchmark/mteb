from mteb.abstasks.any_classification import AbsTaskAnyClassification
from mteb.abstasks.task_metadata import TaskMetadata


class OpenTenderClassification(AbsTaskAnyClassification):
    metadata = TaskMetadata(
        name="OpenTenderClassification",
        dataset={
            "path": "clips/mteb-nl-opentender-cls",
            "revision": "53221b9d10649a531dceccdab8155ab795a59bbb",
        },
        description="This dataset contains Belgian and Dutch tender calls from OpenTender in Dutch",
        reference="https://arxiv.org/abs/2509.12340",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="f1",
        date=("2025-08-01", "2025-08-10"),
        domains=["Government", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{banar2025mtebnle5nlembeddingbenchmark,
  archiveprefix = {arXiv},
  author = {Nikolay Banar and Ehsan Lotfi and Jens Van Nooten and Cristina Arhiliuc and Marija Kliocaite and Walter Daelemans},
  eprint = {2509.12340},
  primaryclass = {cs.CL},
  title = {MTEB-NL and E5-NL: Embedding Benchmark and Models for Dutch},
  url = {https://arxiv.org/abs/2509.12340},
  year = {2025},
}
""",
    )

    def dataset_transform(self) -> None:
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].map(
                lambda ex: {"text": f"{ex['title']}\n{ex['description']}"}
            )

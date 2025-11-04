from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class OpenTenderClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OpenTenderClassification",
        dataset={
            "path": "clips/mteb-nl-opentender-cls-pr",
            "revision": "9af5657575a669dc18c7f897a67287ff7d1a0c65",
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
        prompt={
            "query": "Classificeer de gegeven aanbestedingsbeschrijving in het juiste onderwerp of thema"
        },
    )

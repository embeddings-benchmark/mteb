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
        reference="https://aclanthology.org/2026.findings-acl.1236/",
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
@inproceedings{banar2025mtebnle5nlembeddingbenchmark,
  address = {San Diego, California, United States},
  author = {Banar, Nikolay and Lotfi, Ehsan and Van Nooten, Jens and Arhiliuc, Cristina and Kliocaite, Marija and Daelemans, Walter},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026},
  doi = {10.18653/v1/2026.findings-acl.1236},
  editor = {Liakata, Maria and Moreira, Viviane P. and Zhang, Jiajun and Jurgens, David},
  isbn = {979-8-89176-395-1},
  month = {jul},
  pages = {24684--24709},
  publisher = {Association for Computational Linguistics},
  title = {{MTEB}-{NL} and {E}5-{NL}: Embedding Benchmark and Models for {D}utch},
  url = {https://aclanthology.org/2026.findings-acl.1236/},
  year = {2026},
}
""",
        prompt="Classificeer de gegeven aanbestedingsbeschrijving in het juiste onderwerp of thema",
    )

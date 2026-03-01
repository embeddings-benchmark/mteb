from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class GreekLegalCodeClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="GreekLegalCodeClassification",
        description="Greek Legal Code Dataset for Classification. (subset = chapter)",
        reference="https://arxiv.org/abs/2109.15298",
        dataset={
            "path": "mteb/GreekLegalCodeClassification",
            "revision": "be2d1b5af388af523c24b0022786d03713a3b407",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2021-01-01", "2021-01-01"),
        eval_splits=["validation", "test"],
        eval_langs=["ell-Grek"],
        main_score="accuracy",
        domains=["Legal", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{papaloukas-etal-2021-glc,
  address = {Punta Cana, Dominican Republic},
  author = {Papaloukas, Christos and Chalkidis, Ilias and Athinaios, Konstantinos and Pantazi, Despina-Athanasia and Koubarakis, Manolis},
  booktitle = {Proceedings of the Natural Legal Language Processing Workshop 2021},
  doi = {10.48550/arXiv.2109.15298},
  pages = {63--75},
  publisher = {Association for Computational Linguistics},
  title = {Multi-granular Legal Topic Classification on Greek Legislation},
  url = {https://arxiv.org/abs/2109.15298},
  year = {2021},
}
""",
    )

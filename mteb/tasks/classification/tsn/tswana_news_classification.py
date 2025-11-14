from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class TswanaNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TswanaNewsClassification",
        description="Tswana News Classification Dataset",
        reference="https://link.springer.com/chapter/10.1007/978-3-031-49002-6_17",
        dataset={
            "path": "dsfsi/daily-news-dikgang",
            "revision": "061ca1525717eebaaa9bada240f6cbb31eb3aa87",
        },
        type="Classification",
        task_subtypes=["Topic classification"],
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tsn-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2023-01-01"),
        domains=["News", "Written"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{marivate2023puoberta,
  author = {Vukosi Marivate and Moseli Mots'Oehli and Valencia Wagner and Richard Lastrucci and Isheanesu Dzingirai},
  booktitle = {SACAIR 2023 (To Appear)},
  dataset_url = {https://github.com/dsfsi/PuoBERTa},
  keywords = {NLP},
  preprint_url = {https://arxiv.org/abs/2310.09141},
  software_url = {https://huggingface.co/dsfsi/PuoBERTa},
  title = {PuoBERTa: Training and evaluation of a curated language model for Setswana},
  year = {2023},
}
""",
        superseded_by="TswanaNewsClassification.v2",
    )


class TswanaNewsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TswanaNewsClassification.v2",
        description="Tswana News Classification Dataset This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://link.springer.com/chapter/10.1007/978-3-031-49002-6_17",
        dataset={
            "path": "mteb/tswana_news",
            "revision": "2bbd0687d1733ac419fba18378bd9d864aae081c",
        },
        type="Classification",
        task_subtypes=["Topic classification"],
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tsn-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2023-01-01"),
        domains=["News", "Written"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{marivate2023puoberta,
  author = {Vukosi Marivate and Moseli Mots'Oehli and Valencia Wagner and Richard Lastrucci and Isheanesu Dzingirai},
  booktitle = {SACAIR 2023 (To Appear)},
  dataset_url = {https://github.com/dsfsi/PuoBERTa},
  keywords = {NLP},
  preprint_url = {https://arxiv.org/abs/2310.09141},
  software_url = {https://huggingface.co/dsfsi/PuoBERTa},
  title = {PuoBERTa: Training and evaluation of a curated language model for Setswana},
  year = {2023},
}
""",
        adapted_from=["TswanaNewsClassification"],
    )

from mteb.abstasks import AbsTaskClassification  # type: ignore
from mteb.abstasks.TaskMetadata import TaskMetadata  # type: ignore


class Moroco(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Moroco",
        dataset={
            "path": "universityofbucharest/moroco",
            "revision": "d64d9b8cd876056a5c24552afe3caf7e6fd26c8e",
            "trust_remote_code": True,
        },
        description="The Moldavian and Romanian Dialectal Corpus. The MOROCO data set contains Moldavian and Romanian samples of text collected from the news domain. The samples belong to one of the following six topics: (0) culture, (1) finance, (2) politics, (3) science, (4) sports, (5) tech",
        reference="https://huggingface.co/datasets/moroco",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        date=("2017-10-01", "2017-10-31"),
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="CC BY-4.0",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=[
            "ron-Latn-ron",
            "ron-Latn-mol",
        ],  # Moldavian, or the Romanian dialect used in Moldova, does not have an ISO 639-1 code assigned to it. However, it has been given the three-letter code "mol" under ISO 639-3
        text_creation="found",
        bibtex_citation=""""
        @inproceedings{ Butnaru-ACL-2019,
        author = {Andrei M. Butnaru and Radu Tudor Ionescu},
        title = "{MOROCO: The Moldavian and Romanian Dialectal Corpus}",
        booktitle = {Proceedings of ACL},
        year = {2019},
        pages={688--698},
        }
        """,
        n_samples={"test": 2048},
        avg_character_length={"test": 1710.94},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"sample": "text", "category": "label"}
        ).remove_columns(["id"])
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )

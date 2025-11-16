from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class Moroco(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Moroco",
        dataset={
            "path": "mteb/Moroco",
            "revision": "42c54720ccd5397ed3eede28b2c5ada63a977bf7",
        },
        description="The Moldavian and Romanian Dialectal Corpus. The MOROCO data set contains Moldavian and Romanian samples of text collected from the news domain. The samples belong to one of the following six topics: (0) culture, (1) finance, (2) politics, (3) science, (4) sports, (5) tech",
        reference="https://huggingface.co/datasets/moroco",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        date=("2017-10-01", "2017-10-31"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[
            "ron-Latn-ron",
            "ron-Latn-mol",
        ],  # Moldavian, or the Romanian dialect used in Moldova, does not have an ISO 639-1 code assigned to it. However, it has been given the three-letter code "mol" under ISO 639-3
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Butnaru-ACL-2019,
  author = {Andrei M. Butnaru and Radu Tudor Ionescu},
  booktitle = {Proceedings of ACL},
  pages = {688--698},
  title = {{MOROCO: The Moldavian and Romanian Dialectal Corpus}},
  year = {2019},
}
""",
        superseded_by="Moroco.v2",
    )


class MorocoV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Moroco.v2",
        dataset={
            "path": "mteb/moroco",
            "revision": "6e70588dbd3d583da8b85989c1c3ab3d4bd2e7c4",
        },
        description="The Moldavian and Romanian Dialectal Corpus. The MOROCO data set contains Moldavian and Romanian samples of text collected from the news domain. The samples belong to one of the following six topics: (0) culture, (1) finance, (2) politics, (3) science, (4) sports, (5) tech This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/moroco",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ron-Latn"],
        main_score="accuracy",
        date=("2017-10-01", "2017-10-31"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[
            "ron-Latn-ron",
            "ron-Latn-mol",
        ],  # Moldavian, or the Romanian dialect used in Moldova, does not have an ISO 639-1 code assigned to it. However, it has been given the three-letter code "mol" under ISO 639-3
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Butnaru-ACL-2019,
  author = {Andrei M. Butnaru and Radu Tudor Ionescu},
  booktitle = {Proceedings of ACL},
  pages = {688--698},
  title = {{MOROCO: The Moldavian and Romanian Dialectal Corpus}},
  year = {2019},
}
""",
        adapted_from=["Moroco"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )

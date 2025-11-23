from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class EstonianValenceClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="EstonianValenceClassification",
        dataset={
            "path": "kardosdrur/estonian-valence",
            "revision": "9157397f05a127b3ac93b93dd88abf1bdf710c22",
        },
        description="Dataset containing annotated Estonian news data from the Postimees and Õhtuleht newspapers.",
        reference="https://figshare.com/articles/dataset/Estonian_Valence_Corpus_Eesti_valentsikorpus/24517054",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["est-Latn"],
        main_score="accuracy",
        date=(
            "1857-01-01",  # Inception of Postimees
            "2023-11-08",  # dataset publication
        ),
        domains=["News", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        dialect=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        sample_creation="found",
        bibtex_citation=r"""
@article{Pajupuu2023,
  author = {Hille Pajupuu and Jaan Pajupuu and Rene Altrov and Kairi Tamuri},
  doi = {10.6084/m9.figshare.24517054.v1},
  month = {11},
  title = {{Estonian Valence Corpus  / Eesti valentsikorpus}},
  url = {https://figshare.com/articles/dataset/Estonian_Valence_Corpus_Eesti_valentsikorpus/24517054},
  year = {2023},
}
""",
        superseded_by="EstonianValenceClassification.v2",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("paragraph", "text").rename_column(
            "valence", "label"
        )
        # convert label to a numbers
        labels = self.dataset["train"]["label"]  # type: ignore
        lab2idx = {lab: idx for idx, lab in enumerate(set(labels))}
        self.dataset = self.dataset.map(
            lambda x: {"label": lab2idx[x["label"]]}, remove_columns=["label"]
        )

    samples_per_label = 16


class EstonianValenceClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="EstonianValenceClassification.v2",
        dataset={
            "path": "mteb/estonian_valence",
            "revision": "8795961e2af5b83bcb8a6928636845ac2b92f92e",
        },
        description="Dataset containing annotated Estonian news data from the Postimees and Õhtuleht newspapers. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://figshare.com/articles/dataset/Estonian_Valence_Corpus_Eesti_valentsikorpus/24517054",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["est-Latn"],
        main_score="accuracy",
        date=(
            "1857-01-01",  # Inception of Postimees
            "2023-11-08",  # dataset publication
        ),
        domains=["News", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        dialect=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        sample_creation="found",
        bibtex_citation=r"""
@article{Pajupuu2023,
  author = {Hille Pajupuu and Jaan Pajupuu and Rene Altrov and Kairi Tamuri},
  doi = {10.6084/m9.figshare.24517054.v1},
  month = {11},
  title = {{Estonian Valence Corpus  / Eesti valentsikorpus}},
  url = {https://figshare.com/articles/dataset/Estonian_Valence_Corpus_Eesti_valentsikorpus/24517054},
  year = {2023},
}
""",
        adapted_from=["EstonianValenceClassification"],
    )

    samples_per_label = 16

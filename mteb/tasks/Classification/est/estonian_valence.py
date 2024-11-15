from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


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
        category="s2s",
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
        bibtex_citation="""
@article{Pajupuu2023,
    author = "Hille Pajupuu and Jaan Pajupuu and Rene Altrov and Kairi Tamuri",
    title = "{Estonian Valence Corpus  / Eesti valentsikorpus}",
    year = "2023",
    month = "11",
    url = "https://figshare.com/articles/dataset/Estonian_Valence_Corpus_Eesti_valentsikorpus/24517054",
    doi = "10.6084/m9.figshare.24517054.v1"
}""",
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

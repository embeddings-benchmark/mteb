from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
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
        eval_splits=["test"],
        eval_langs=["est-Latn"],
        main_score="accuracy",
        date=(
            "1857-01-01",  # Inception of Postimees
            "2023-11-08",  # dataset publication
        ),
        form=["written"],
        domains=["News"],
        task_subtypes=["Sentiment/Hate speech"],
        dialect=[],
        license="CC BY 4.0",
        socioeconomic_status="high",
        annotations_creators="human-annotated",
        text_creation="found",
        bibtex_citation="""
@article{Pajupuu2023,
    author = "Hille Pajupuu and Jaan Pajupuu and Rene Altrov and Kairi Tamuri",
    title = "{Estonian Valence Corpus  / Eesti valentsikorpus}",
    year = "2023",
    month = "11",
    url = "https://figshare.com/articles/dataset/Estonian_Valence_Corpus_Eesti_valentsikorpus/24517054",
    doi = "10.6084/m9.figshare.24517054.v1"
}""",
        n_samples={"train": 3270, "test": 818},
        avg_character_length={"train": 226.70642201834863, "test": 231.5085574572127},
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

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 16
        return metadata_dict

from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class KannadaNewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KannadaNewsClassification",
        description="The Kannada news dataset contains only the headlines of news article in three categories: Entertainment, Tech, and Sports. The data set contains around 6300 news article headlines which are collected from Kannada news websites. The data set has been cleaned and contains train and test set using which can be used to benchmark topic classification models in Kannada.",
        dataset={
            "path": "Akash190104/kannada_news_classification",
            "revision": "a470711069906ac0a559defec3b89cb3725601bd",
        },
        reference="https://github.com/goru001/nlp-for-kannada",
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["kan-Knda"],
        main_score="accuracy",
        date=("2019-03-17", "2020-08-06"),
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="CC-BY-SA-4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples={"train": 6460},
        avg_character_length={"train": 65.88},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("headline", "text")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )

from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class Kannada_News_Classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Kannada_News_Classification",
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
        date=None,
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation="found",
        bibtex_citation=None,
        n_samples={"train": 6460},
        avg_character_length={"train": 65.88},
    )

    def dataset_transform(self):
        self.dataset["train"] = self.dataset["train"].select(range(2048))
        self.dataset = self.dataset.rename_column("headline", "text")

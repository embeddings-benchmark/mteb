from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class YahooAnswersTopicsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="YahooAnswersTopicsClassification",
        description="Dataset composed of questions and answers from Yahoo Answers, categorized into topics.",
        reference="https://huggingface.co/datasets/yahoo_answers_topics",
        dataset={
            "path": "yahoo_answers_topics",
            "revision": "78fccffa043240c80e17a6b1da724f5a1057e8e5",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-01-25", "2022-01-25"),
        form=["written"],
        domains=["Web"],
        task_subtypes=["Topic Classification"],
        license="Not specified",
        socioeconomic_status="low",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples={"test": 60000},
        avg_character_length={"test": 346.35},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 32
        return metadata_dict

    def dataset_transform(self):
        self.dataset = self.dataset.map(
            lambda examples: examples,
            remove_columns=["id", "question_title", "question_content"],
        )

        self.dataset = self.dataset.rename_columns(
            {"topic": "label", "best_answer": "text"}
        )

        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train", "test"]
        )

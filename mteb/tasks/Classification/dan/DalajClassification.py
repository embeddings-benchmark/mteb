# SuperLIM tasks
from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class DalajClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DalajClassification",
        dataset={
            "path": "AI-Sweden/SuperLim",
            "revision": "7ebf0b4caa7b2ae39698a889de782c09e6f5ee56",
            "name": "dalaj",
        },
        description="A Swedish dataset for linguistic acceptability. Available as a part of Superlim.",
        reference="https://spraakbanken.gu.se/en/resources/superlim",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="accuracy",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 444},
        avg_character_length={"test": 243.8},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 16
        return metadata_dict

    def dataset_transform(self):
        """This dataset consist of two columns of relevance, "original_sentence" and "corrected_sentence".
        We will use the original sentence as we "wrong" sentence and the corrected sentence as the "correct" sentence
        """

        def __convert_sample_to_classification(sample):
            text = sample["original_sentence"] + sample["corrected_sentence"]
            label = [1] * len(sample["original_sentence"]) + [0] * len(
                sample["corrected_sentence"]
            )
            return {"text": text, "label": label}

        columns_to_keep = ["original_sentence", "corrected_sentence"]
        for split in self.dataset:
            columns_names = self.dataset[split].column_names  # type: ignore
            columns_to_remove = [
                col for col in columns_names if col not in columns_to_keep
            ]
            self.dataset[split] = self.dataset[split].remove_columns(columns_to_remove)  # type: ignore

        self.dataset = self.dataset.map(
            __convert_sample_to_classification,
            batched=True,
            remove_columns=columns_to_keep,
        )

from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class FinancialPhrasebankClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinancialPhrasebankClassification",
        description="Polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral.",
        reference="https://arxiv.org/abs/1307.5336",
        dataset={
            "path": "financial_phrasebank",
            "revision": "1484d06fe7af23030c7c977b12556108d1f67039",
            "name": "sentences_allagree",
        },
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=None,
        form=None,
        domains=["News"],
        task_subtypes=None,
        license="cc-by-nc-sa-3.0",
        socioeconomic_status=None,
        annotations_creators="expert-annotated",
        dialect=None,
        text_creation=None,
        bibtex_citation="""
            @article{Malo2014GoodDO,
            title={Good debt or bad debt: Detecting semantic orientations in economic texts},
            author={P. Malo and A. Sinha and P. Korhonen and J. Wallenius and P. Takala},
            journal={Journal of the Association for Information Science and Technology},
            year={2014},
            volume={65}
            }
        """,
        n_samples={"train": 4840},
        avg_character_length=None,  # Optional: You can calculate this based on the dataset if needed
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 128
        return metadata_dict

    def dataset_transform(self):
        self.dataset = self.dataset.map(
            lambda examples: {"text": examples["sentence"]}, remove_columns=["sentence"]
        )

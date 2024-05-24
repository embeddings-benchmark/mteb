from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class FinancialPhrasebankClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinancialPhrasebankClassification",
        description="Polar sentiment dataset of sentences from financial news, categorized by sentiment into positive, negative, or neutral.",
        reference="https://arxiv.org/abs/1307.5336",
        dataset={
            "path": "takala/financial_phrasebank",
            "revision": "1484d06fe7af23030c7c977b12556108d1f67039",
            "name": "sentences_allagree",
        },
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2013-11-01", "2013-11-01"),
        form=["written"],
        domains=["News"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-nc-sa-3.0",
        socioeconomic_status="medium",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
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
        avg_character_length={"train": 121.96},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")

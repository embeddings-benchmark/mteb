from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class BulgarianStoreReviewSentimentClassfication(AbsTaskClassification):
    metadata = TaskMetadata(
        name="BulgarianStoreReviewSentimentClassfication",
        description="Bulgarian online store review dataset for sentiment classification.",
        reference="https://doi.org/10.7910/DVN/TXIK9P",
        dataset={
            "path": "artist/Bulgarian-Online-Store-Feedback-Text-Analysis",
            "revision": "701984d6c6efea0e14a1c7850ef70e464c5577c0",
        },
        type="Classification",
        category="s2s",
        date=("2018-05-14", "2018-05-14"),
        eval_splits=["test"],
        eval_langs=["bul-Cyrl"],
        main_score="accuracy",
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@data{DVN/TXIK9P_2018,
author = {Georgieva-Trifonova, Tsvetanka and Stefanova, Milena and Kalchev, Stefan},
publisher = {Harvard Dataverse},
title = {{Dataset for ``Customer Feedback Text Analysis for Online Stores Reviews in Bulgarian''}},
year = {2018},
version = {V1},
doi = {10.7910/DVN/TXIK9P},
url = {https://doi.org/10.7910/DVN/TXIK9P}
}
""",
        n_samples={"test": 182},
        avg_character_length={"test": 316.7},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"Review": "text", "Category": "label"}
        )

        labels = self.dataset["train"]["label"]
        lab2idx = {lab: idx for idx, lab in enumerate(sorted(set(labels)))}

        self.dataset = self.dataset.map(
            lambda x: {"label": lab2idx[x["label"]]}, remove_columns=["label"]
        )

from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.abs_text_classification import AbsTextClassification


class GujaratiNewsClassification(AbsTextClassification):
    metadata = TaskMetadata(
        name="GujaratiNewsClassification",
        description="A Gujarati dataset for 3-class classification of Gujarati news articles",
        reference="https://github.com/goru001/nlp-for-gujarati",
        dataset={
            "path": "mlexplorer008/gujarati_news_classification",
            "revision": "1a5f2fa2914bfeff4fcdc6fff4194fa8ec8fa19e",
        },
        type="Classification",
        category="t2t",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["guj-Gujr"],
        main_score="accuracy",
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",  # none found
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("headline", "text")

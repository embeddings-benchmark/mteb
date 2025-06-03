from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class GermanPoliticiansTwitterSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="GermanPoliticiansTwitterSentimentClassification",
        description="GermanPoliticiansTwitterSentiment is a dataset of German tweets categorized with their sentiment (3 classes).",
        reference="https://aclanthology.org/2022.konvens-1.9",
        dataset={
            "path": "Alienmaster/german_politicians_twitter_sentiment",
            "revision": "65343b17f5a76227ab2e15b9424dfab6466ffcb1",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2021-12-31"),
        domains=["Social", "Government", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{schmidt-etal-2022-sentiment,
  address = {Potsdam, Germany},
  author = {Schmidt, Thomas  and
Fehle, Jakob  and
Weissenbacher, Maximilian  and
Richter, Jonathan  and
Gottschalk, Philipp  and
Wolff, Christian},
  booktitle = {Proceedings of the 18th Conference on Natural Language Processing (KONVENS 2022)},
  editor = {Schaefer, Robin  and
Bai, Xiaoyu  and
Stede, Manfred  and
Zesch, Torsten},
  month = {12--15 } # sep,
  pages = {74--87},
  publisher = {KONVENS 2022 Organizers},
  title = {Sentiment Analysis on {T}witter for the Major {G}erman Parties during the 2021 {G}erman Federal Election},
  url = {https://aclanthology.org/2022.konvens-1.9},
  year = {2022},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("majority_sentiment", "label")

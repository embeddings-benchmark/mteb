from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class RuReviewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuReviewsClassification",
        dataset={
            "path": "ai-forever/ru-reviews-classification",
            "revision": "f6d2c31f4dc6b88f468552750bfec05b4b41b05a",
        },
        description="Product review classification (3-point scale) based on RuRevies dataset",
        reference="https://github.com/sismetanin/rureviews",
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("2000-01-01", "2020-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@INPROCEEDINGS{Smetanin-SA-2019,
        author={Sergey Smetanin and Michail Komarov},
        booktitle={2019 IEEE 21st Conference on Business Informatics (CBI)},
        title={Sentiment Analysis of Product Reviews in Russian using Convolutional Neural Networks},
        year={2019},
        volume={01},
        number={},
        pages={482-486},
        doi={10.1109/CBI.2019.00062},
        ISSN={2378-1963},
        month={July}
        }""",
        prompt="Classify product reviews into positive, negative or neutral sentiment",
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, n_samples=2048, splits=["test"]
        )

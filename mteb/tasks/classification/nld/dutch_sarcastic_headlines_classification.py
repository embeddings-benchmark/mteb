from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class DutchSarcasticHeadlinesClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DutchSarcasticHeadlinesClassification",
        description="This dataset contains news headlines of two Dutch news websites. All sarcastic headlines were "
        "collected from the Speld.nl (the Dutch equivalent of The Onion) whereas all 'normal' headlines "
        "were collected from the news website Nu.nl.",
        reference="https://www.kaggle.com/datasets/harrotuin/dutch-news-headlines",
        dataset={
            "path": "clips/mteb-nl-sarcastic-headlines",
            "revision": "7e520e36394795859583f84f81fcb97de915d05a",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2019-01-01", "2020-01-01"),
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="f1",
        domains=["News", "Written", "Fiction"],
        task_subtypes=[],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""""",
        prompt={
            "query": "Classificeer de gegeven krantenkop als sarcastisch of niet sarcastisch"
        },
    )

    def dataset_transform(self):
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].rename_columns(
                {"headline": "text", "is_sarcastic": "label"}
            )

from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification

# type: ignore
from mteb.abstasks.TaskMetadata import TaskMetadata  # type: ignore


class HebrewSentimentAnalysis(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HebrewSentimentAnalysis",
        dataset={
            "path": "omilab/hebrew_sentiment",
            "revision": "952c9525954c1dac50d5f95945eb5585bb6464e7",
            "name": "morph",
            "trust_remote_code": True,
        },
        description="HebrewSentiment is a data set consists of 12,804 user comments to posts on the official Facebook page of Israel’s president, Mr. Reuven Rivlin. In October 2015, we used the open software application Netvizz (Rieder, 2013) to scrape all the comments to all of the president’s posts in the period of June – August 2014, the first three months of Rivlin’s presidency.2 While the president’s posts aimed at reconciling tensions and called for tolerance and empathy, the sentiment expressed in the comments to the president’s posts was polarized between citizens who warmly thanked the president, and citizens that fiercely critiqued his policy.",
        reference="https://huggingface.co/datasets/hebrew_sentiment",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["heb-Hebr"],
        main_score="accuracy",
        date=("2015-10-01", "2015-10-31"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""""
        @inproceedings{amram-etal-2018-representations,
            title = "Representations and Architectures in Neural Sentiment Analysis for Morphologically Rich Languages: A Case Study from {M}odern {H}ebrew",
            author = "Amram, Adam and Ben David, Anat and Tsarfaty, Reut",
            booktitle = "Proceedings of the 27th International Conference on Computational Linguistics",
            month = aug,
            year = "2018",
            address = "Santa Fe, New Mexico, USA",
            publisher = "Association for Computational Linguistics",
            url = "https://www.aclweb.org/anthology/C18-1190",
            pages = "2242--2252"
        }
        """,
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )

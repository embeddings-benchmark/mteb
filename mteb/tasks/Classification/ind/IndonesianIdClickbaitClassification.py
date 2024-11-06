from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class IndonesianIdClickbaitClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="IndonesianIdClickbaitClassification",
        dataset={
            "path": "manandey/id_clickbait",
            "revision": "9fa4d0824015fe537ae2c8166781f5c79873da2c",
        },
        description="The CLICK-ID dataset is a collection of Indonesian news headlines that was collected from 12 local online news publishers.",
        reference="http://www.sciencedirect.com/science/article/pii/S2352340920311252",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["ind-Latn"],
        main_score="f1",
        date=("2020-10-01", "2020-10-01"),
        domains=["News", "Written"],
        dialect=[],
        task_subtypes=["Claim verification"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        sample_creation="found",
        bibtex_citation="""
        @article{WILLIAM2020106231,
title = "CLICK-ID: A novel dataset for Indonesian clickbait headlines",
journal = "Data in Brief",
volume = "32",
pages = "106231",
year = "2020",
issn = "2352-3409",
doi = "https://doi.org/10.1016/j.dib.2020.106231",
url = "http://www.sciencedirect.com/science/article/pii/S2352340920311252",
author = "Andika William and Yunita Sari",
keywords = "Indonesian, Natural Language Processing, News articles, Clickbait, Text-classification",
abstract = "News analysis is a popular task in Natural Language Processing (NLP). In particular, the problem of clickbait in news analysis has gained attention in recent years [1, 2]. However, the majority of the tasks has been focused on English news, in which there is already a rich representative resource. For other languages, such as Indonesian, there is still a lack of resource for clickbait tasks. Therefore, we introduce the CLICK-ID dataset of Indonesian news headlines extracted from 12 Indonesian online news publishers. It is comprised of 15,000 annotated headlines with clickbait and non-clickbait labels. Using the CLICK-ID dataset, we then developed an Indonesian clickbait classification model achieving favourable performance. We believe that this corpus will be useful for replicable experiments in clickbait detection or other experiments in NLP areas."
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.remove_columns(["label"]).rename_columns(
            {"title": "text", "label_score": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["train"]
        )

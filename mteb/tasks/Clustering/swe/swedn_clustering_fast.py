from __future__ import annotations

import datasets

from mteb.abstasks import TaskMetadata
from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast


def dataset_transform(self):
    """The article_category clusters differ between the splits (with the test set only having 1 cluster). Therefore we combine it all into one
    cluster.
    """
    splits = ["train", "validation"]
    # performance of sample models with test set: 8.74, 2.43 -removing test-> 11.26, 4.27
    # this is due to the test set only having 1 cluster which is "other"

    label_col = "article_category"

    labels_headlines = []
    labels_summaries = []
    labels_articles = []
    docs_headlines = []
    docs_summaries = []
    docs_articles = []

    for split in splits:
        ds_split = self.dataset[split]

        docs_headlines.extend(ds_split["headline"])
        labels_headlines.extend(ds_split[label_col])

        docs_summaries.extend(ds_split["summary"])
        labels_summaries.extend(ds_split[label_col])

        docs_articles.extend(ds_split["article"])
        labels_articles.extend(ds_split[label_col])

    ds_headlines = datasets.Dataset.from_dict(
        {"sentences": docs_headlines, "labels": labels_headlines}
    )
    ds_summaries = datasets.Dataset.from_dict(
        {"sentences": docs_summaries, "labels": labels_summaries}
    )
    ds_articles = datasets.Dataset.from_dict(
        {"sentences": docs_articles, "labels": labels_articles}
    )

    self.dataset = datasets.DatasetDict(
        {
            "headlines": ds_headlines,
            "summaries": ds_summaries,
            "articles": ds_articles,
        }
    )


class SwednClusteringP2P(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="SwednClusteringP2P",
        dataset={
            "path": "sbx/superlim-2",
            "revision": "ef1661775d746e0844b299164773db733bdc0bf6",
            "name": "swedn",
            "trust_remote_code": True,
        },
        description="The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure. This dataset uses the category labels as clusters.",
        reference="https://spraakbanken.gu.se/en/resources/swedn",
        type="Clustering",
        category="p2p",
        eval_splits=["summaries", "articles"],
        eval_langs=["swe-Latn"],
        main_score="v_measure",
        date=("2000-01-01", "2020-12-31"),  # best guess
        form=["written"],
        domains=["News", "Non-fiction"],
        license=None,
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Thematic clustering"],
        text_creation="found",
        bibtex_citation="""@inproceedings{monsen2021method,
  title={A method for building non-english corpora for abstractive text summarization},
  author={Monsen, Julius and J{\"o}nsson, Arne},
  booktitle={Proceedings of CLARIN Annual Conference},
  year={2021}
}""",
        n_samples={"all": 2048},
        avg_character_length={"all": 1619.71},
    )

    def dataset_transform(self):
        dataset_transform(self)


class SwednClusteringFastS2S(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="SwednClusteringS2S",
        dataset={
            "path": "sbx/superlim-2",
            "revision": "ef1661775d746e0844b299164773db733bdc0bf6",
            "name": "swedn",
            "trust_remote_code": True,
        },
        description="The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure. This dataset uses the category labels as clusters.",
        reference="https://spraakbanken.gu.se/en/resources/swedn",
        type="Clustering",
        category="s2s",
        eval_splits=["headlines"],
        eval_langs=["swe-Latn"],
        main_score="v_measure",
        date=("2000-01-01", "2020-12-31"),  # best guess
        form=["written"],
        domains=["News", "Non-fiction"],
        license=None,
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Thematic clustering"],
        text_creation="found",
        bibtex_citation="""@inproceedings{monsen2021method,
  title={A method for building non-english corpora for abstractive text summarization},
  author={Monsen, Julius and J{\"o}nsson, Arne},
  booktitle={Proceedings of CLARIN Annual Conference},
  year={2021}
}""",
        n_samples={"all": 2048},
        avg_character_length={"all": 1619.71},
    )

    def dataset_transform(self):
        dataset_transform(self)

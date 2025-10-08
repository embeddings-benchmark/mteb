from __future__ import annotations

from mteb.abstasks.AbsTaskAnyClustering import AbsTaskAnyClustering
from mteb.abstasks.task_metadata import TaskMetadata


class SwednClustering(AbsTaskAnyClustering):
    superseded_by = "SwednClusteringP2P"

    metadata = TaskMetadata(
        name="SwednClustering",
        dataset={
            "path": "mteb/SwednClustering",
            "revision": "7125017ead5797297f46e17b31bf78b56d12c2b2",
        },
        description="The SWE-DN corpus is based on 1,963,576 news articles from the Swedish newspaper Dagens Nyheter (DN) during the years 2000--2020. The articles are filtered to resemble the CNN/DailyMail dataset both regarding textual structure. This dataset uses the category labels as clusters.",
        reference="https://spraakbanken.gu.se/en/resources/swedn",
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["all"],
        eval_langs=["swe-Latn"],
        main_score="v_measure",
        date=("2000-01-01", "2020-12-31"),  # best guess
        domains=["News", "Non-fiction", "Written"],
        license=None,
        annotations_creators="derived",
        dialect=[],
        task_subtypes=["Thematic clustering"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{monsen2021method,
  author = {Monsen, Julius and J{\"o}nsson, Arne},
  booktitle = {Proceedings of CLARIN Annual Conference},
  title = {A method for building non-english corpora for abstractive text summarization},
  year = {2021},
}
""",
    )

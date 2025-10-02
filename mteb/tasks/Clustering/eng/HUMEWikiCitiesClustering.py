from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class HUMEWikiCitiesClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="HUMEWikiCitiesClustering",
        description="Human evaluation subset of Clustering of Wikipedia articles of cities by country from https://huggingface.co/datasets/wikipedia. Test set includes 126 countries, and a total of 3531 cities.",
        reference="https://huggingface.co/datasets/wikipedia",
        dataset={
            "path": "mteb/mteb-human-wikicities-clustering",
            "revision": "5c46af681d2dfa6d3ee373b7ccb4f153e1b72792",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2000-01-01", "2021-12-31"),  # very rough estimate
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@online{wikidump,
  author = {Wikimedia Foundation},
  title = {Wikimedia Downloads},
  url = {https://dumps.wikimedia.org},
}
""",
        adapted_from=["WikiCitiesClustering"],
    )

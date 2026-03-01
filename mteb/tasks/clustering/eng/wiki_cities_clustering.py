from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class WikiCitiesClustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="WikiCitiesClustering",
        description="Clustering of Wikipedia articles of cities by country from https://huggingface.co/datasets/wikipedia. Test set includes 126 countries, and a total of 3531 cities.",
        reference="https://huggingface.co/datasets/wikipedia",
        dataset={
            "path": "mteb/WikiCitiesClustering",
            "revision": "9f302fc86ddf2d9133ebcc03ee3a285f4729bb16",
        },
        type="Clustering",
        category="t2c",
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
@online{wikidump2024,
  author = {Wikimedia Foundation},
  title = {Wikimedia Downloads},
  url = {https://dumps.wikimedia.org},
}
""",
    )

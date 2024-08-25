from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class WikiCitiesClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="WikiCitiesClustering",
        description="Clustering of Wikipedia articles of cities by country from https://huggingface.co/datasets/wikipedia. Test set includes 126 countries, and a total of 3531 cities.",
        reference="https://huggingface.co/datasets/wikipedia",
        dataset={
            "path": "jinaai/cities_wiki_clustering",
            "revision": "ddc9ee9242fa65332597f70e967ecc38b9d734fa",
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
        bibtex_citation="""@ONLINE{wikidump,
    author = "Wikimedia Foundation",
    title  = "Wikimedia Downloads",
    url    = "https://dumps.wikimedia.org"
}""",
        descriptive_stats={"n_samples": None, "avg_character_length": None},
    )

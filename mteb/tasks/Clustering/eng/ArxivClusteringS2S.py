from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class ArxivClusteringS2S(AbsTaskClustering):
    superseded_by = "ArXivHierarchicalClusteringS2S"
    metadata = TaskMetadata(
        name="ArxivClusteringS2S",
        description="Clustering of titles from arxiv. Clustering of 30 sets, either on the main or secondary category",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={
            "path": "mteb/arxiv-clustering-s2s",
            "revision": "f910caf1a6075f7329cdf8c1a6135696f37dbd53",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{arxiv_org_submitters_2024,
    title={arXiv Dataset},
    url={https://www.kaggle.com/dsv/7548853},
    DOI={10.34740/KAGGLE/DSV/7548853},
    publisher={Kaggle},
    author={arXiv.org submitters},
    year={2024}
}""",
        descriptive_stats={
            "n_samples": {"test": 732723},
            "avg_character_length": {"test": 74},
        },
    )

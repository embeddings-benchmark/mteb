from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class BuiltBenchClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="BuiltBenchClusteringP2P",
        description="Clustering of built asset item descriptions based on categories identified within industry classification systems such as IFC, Uniclass, etc.",
        reference="https://arxiv.org/abs/2411.12056",
        dataset={
            "path": "mehrzad-shahin/BuiltBench-clustering-p2p",
            "revision": "919bb71053e9de62a68998161ce4f0cee8f786fb",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2024-06-01", "2024-11-30"),
        domains=["Engineering", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-nd-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@article{shahinmoghadam2024benchmarking,
    title={Benchmarking pre-trained text embedding models in aligning built asset information},
    author={Shahinmoghadam, Mehrzad and Motamedi, Ali},
    journal={arXiv preprint arXiv:2411.12056},
    year={2024}
}""",
        prompt="Identify the category of the built asset entities based on the entity description",
    )

from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class BuiltBenchClusteringS2S(AbsTaskClustering):
    metadata = TaskMetadata(
        name="BuiltBenchClusteringS2S",
        description="Clustering of built asset names/titles based on categories identified within industry classification systems such as IFC, Uniclass, etc.",
        reference="https://arxiv.org/abs/2411.12056",
        dataset={
            "path": "mehrzad-shahin/BuiltBench-clustering-s2s",
            "revision": "1aaeb2ece89ea0a8c64e215c95c4cfaf7e891149",
        },
        type="Clustering",
        category="s2s",
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
        bibtex_citation=r"""
@article{shahinmoghadam2024benchmarking,
  author = {Shahinmoghadam, Mehrzad and Motamedi, Ali},
  journal = {arXiv preprint arXiv:2411.12056},
  title = {Benchmarking pre-trained text embedding models in aligning built asset information},
  year = {2024},
}
""",
        prompt="Identify the category of the built asset entities based on the names or titles",
    )

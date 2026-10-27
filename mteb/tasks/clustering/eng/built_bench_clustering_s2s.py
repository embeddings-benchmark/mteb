from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class BuiltBenchClusteringS2S(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="BuiltBenchClusteringS2S",
        description="Clustering of built asset names/titles based on categories identified within industry classification systems such as IFC, Uniclass, etc.",
        reference="https://doi.org/10.1038/s41598-025-09052-5",
        dataset={
            "path": "mehrzad-shahin/BuiltBench-clustering-s2s",
            "revision": "1aaeb2ece89ea0a8c64e215c95c4cfaf7e891149",
        },
        type="Clustering",
        category="t2c",
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
  doi = {10.1038/s41598-025-09052-5},
  issn = {2045-2322},
  journal = {Scientific Reports},
  month = {July},
  number = {1},
  publisher = {Springer Science and Business Media LLC},
  title = {Benchmarking pre-trained text embedding models in aligning built asset information},
  url = {https://doi.org/10.1038/s41598-025-09052-5},
  volume = {15},
  year = {2025},
}
""",
        prompt="Identify the category of the built asset entities based on the names or titles",
    )

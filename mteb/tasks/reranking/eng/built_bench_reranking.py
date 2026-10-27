from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class BuiltBenchReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BuiltBenchReranking",
        description="Reranking of built asset entity type/class descriptions given a query describing an entity as represented in well-established industry classification systems such as Uniclass, IFC, etc.",
        reference="https://doi.org/10.1038/s41598-025-09052-5",
        dataset={
            "path": "mteb/BuiltBenchReranking",
            "revision": "63343e96ffc63a82470fc5bf919ad13951df6c15",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_1000",
        date=("2024-06-01", "2024-11-30"),
        domains=["Engineering", "Written"],
        task_subtypes=[],
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
        prompt={
            "query": "Given a query, retrieve relevant entity descriptions from buit asset classification systems such as IFC and Uniclass"
        },
    )

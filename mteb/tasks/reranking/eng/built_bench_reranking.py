from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class BuiltBenchReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BuiltBenchReranking",
        description="Reranking of built asset entity type/class descriptions given a query describing an entity as represented in well-established industry classification systems such as Uniclass, IFC, etc.",
        reference="https://arxiv.org/abs/2411.12056",
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
  journal = {arXiv preprint arXiv:2411.12056},
  title = {Benchmarking pre-trained text embedding models in aligning built asset information},
  year = {2024},
}
""",
        prompt={
            "query": "Given a query, retrieve relevant entity descriptions from buit asset classification systems such as IFC and Uniclass"
        },
    )

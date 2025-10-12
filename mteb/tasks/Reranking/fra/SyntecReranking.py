from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SyntecReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SyntecReranking",
        description="This dataset has been built from the Syntec Collective bargaining agreement.",
        reference="https://huggingface.co/datasets/lyon-nlp/mteb-fr-reranking-syntec-s2p",
        dataset={
            "path": "mteb/SyntecReranking",
            "revision": "fd3b5633e0e2fec4b744e1d0d6c8bade30ef147e",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="map_at_1000",
        date=("2022-12-01", "2022-12-02"),
        domains=["Legal", "Written"],
        task_subtypes=None,
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=None,
        sample_creation="found",
        bibtex_citation=r"""
@misc{ciancone2024extending,
  archiveprefix = {arXiv},
  author = {Mathieu Ciancone and Imene Kerboua and Marion Schaeffer and Wissam Siblini},
  eprint = {2405.20468},
  primaryclass = {cs.CL},
  title = {Extending the Massive Text Embedding Benchmark to French},
  year = {2024},
}
""",
    )

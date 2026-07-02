from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SkQuadReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SkQuadReranking",
        description=""" From Retrieval Sk QUAD """,
        reference="https://huggingface.co/datasets/TUKE-KEMT/reranking-skquad",
        dataset={
            "path": "TUKE-KEMT/reranking-skquad",
            "revision": "3997eba1c60721e34f7b01db400f5b14a40218ea",
        },
        type="Reranking",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="map_at_1000",
        date=("2025-10-09", "2025-10-09"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        domains=["Encyclopaedic"],
        task_subtypes=["Article retrieval"],
        bibtex_citation=r"""
@article{hladek2023slovak,
  author = {Hl{\'a}dek, Daniel and Sta{\v{s}}, J{\'a}n and Juh{\'a}r, Jozef and Koct{\'u}r, Tom{\'a}{\v{s}}},
  journal = {IEEE Access},
  pages = {32869--32881},
  publisher = {IEEE},
  title = {Slovak dataset for multilingual question answering},
  volume = {11},
  year = {2023},
}
""",
        prompt="Given a query, rerank the documents by their relevance to the query",
    )

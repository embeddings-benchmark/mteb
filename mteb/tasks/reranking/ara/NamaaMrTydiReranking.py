from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NamaaMrTydiReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NamaaMrTydiReranking",
        description="Mr. TyDi is a multi-lingual benchmark dataset built on TyDi, covering eleven typologically diverse languages. It is designed for monolingual retrieval, specifically to evaluate ranking with learned dense representations. This dataset adapts the arabic test split for Reranking evaluation purposes by the addition of multiple (Hard) Negatives to each query and positive",
        reference="https://huggingface.co/NAMAA-Space",
        dataset={
            "path": "mteb/NamaaMrTydiReranking",
            "revision": "4d574b8caf8463c741b84a293aea8ace67801cdc",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ara-Arab"],
        main_score="map_at_1000",
        date=("2023-11-01", "2024-05-15"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="cc-by-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{muennighoff2022mteb,
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\\i}c and Reimers, Nils},
  doi = {10.48550/ARXIV.2210.07316},
  journal = {arXiv preprint arXiv:2210.07316},
  publisher = {arXiv},
  title = {MTEB: Massive Text Embedding Benchmark},
  url = {https://arxiv.org/abs/2210.07316},
  year = {2022},
}
""",
    )

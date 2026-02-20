from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "fas": ["fas-Arab"],
    "rus": ["rus-Cyrl"],
    "zho": ["zho-Hans"],
}


class NeuCLIR2022Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NeuCLIR2022Retrieval",
        description="The task involves identifying and retrieving the documents that are relevant to the queries.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/NeuCLIR2022Retrieval",
            "revision": "95cad8671c1c31908766a689755c307f4770411f",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_20",
        date=("2021-08-01", "2022-06-30"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="odc-by",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{lawrie2023overview,
  author = {Lawrie, Dawn and MacAvaney, Sean and Mayfield, James and McNamee, Paul and Oard, Douglas W and Soldaini, Luca and Yang, Eugene},
  journal = {arXiv preprint arXiv:2304.12367},
  title = {Overview of the TREC 2022 NeuCLIR track},
  year = {2023},
}
""",
    )


class NeuCLIR2022RetrievalHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NeuCLIR2022RetrievalHardNegatives",
        description="The task involves identifying and retrieving the documents that are relevant to the queries. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/NeuCLIR2022RetrievalHardNegatives",
            "revision": "cc5a0360c8bdf1e4afa19d2ea8ec111c0ca96335",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_20",
        date=("2021-08-01", "2022-06-30"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="odc-by",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{lawrie2023overview,
  author = {Lawrie, Dawn and MacAvaney, Sean and Mayfield, James and McNamee, Paul and Oard, Douglas W and Soldaini, Luca and Yang, Eugene},
  journal = {arXiv preprint arXiv:2304.12367},
  title = {Overview of the TREC 2022 NeuCLIR track},
  year = {2023},
}
""",
        adapted_from=["NeuCLIR2022Retrieval"],
    )

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "fas": ["fas-Arab"],
    "rus": ["rus-Cyrl"],
    "zho": ["zho-Hans"],
}


class NeuCLIR2023Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NeuCLIR2023Retrieval",
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
        date=("2022-08-01", "2023-06-30"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="odc-by",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{lawrie2024overview,
  archiveprefix = {arXiv},
  author = {Dawn Lawrie and Sean MacAvaney and James Mayfield and Paul McNamee and Douglas W. Oard and Luca Soldaini and Eugene Yang},
  eprint = {2404.08071},
  primaryclass = {cs.IR},
  title = {Overview of the TREC 2023 NeuCLIR Track},
  year = {2024},
}
""",
    )


class NeuCLIR2023RetrievalHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NeuCLIR2023RetrievalHardNegatives",
        description="The task involves identifying and retrieving the documents that are relevant to the queries. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/NeuCLIR2023RetrievalHardNegatives",
            "revision": "efb44bb789b9f82223aa819e18dba3748fd09e33",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_20",
        date=("2022-08-01", "2023-06-30"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="odc-by",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{lawrie2024overview,
  archiveprefix = {arXiv},
  author = {Dawn Lawrie and Sean MacAvaney and James Mayfield and Paul McNamee and Douglas W. Oard and Luca Soldaini and Eugene Yang},
  eprint = {2404.08071},
  primaryclass = {cs.IR},
  title = {Overview of the TREC 2023 NeuCLIR Track},
  year = {2024},
}
""",
        adapted_from=["NeuCLIR2022Retrieval"],
    )

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class VABBRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VABBRetrieval",
        description="This dataset contains the fourteenth edition of the Flemish Academic Bibliography for the Social "
        "Sciences and Humanities (VABB-SHW), a database of academic publications from the social sciences "
        "and humanities authored by researchers affiliated to Flemish universities (more information). "
        "Publications in the database are used as one of the parameters of the Flemish performance-based "
        "research funding system",
        reference="https://zenodo.org/records/14214806",
        dataset={
            "path": "clips/mteb-nl-vabb-ret",
            "revision": "af4a1e5b3ed451103894f86ff6b3ce85085d7b48",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2009-11-01", "2010-01-01"),
        domains=["Academic", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@dataset{aspeslagh2024vabb,
  author = {Aspeslagh, Pieter and Guns, Raf and Engels, Tim C. E.},
  doi = {10.5281/zenodo.14214806},
  publisher = {Zenodo},
  title = {VABB-SHW: Dataset of Flemish Academic Bibliography for the Social Sciences and Humanities (edition 14)},
  url = {https://doi.org/10.5281/zenodo.14214806},
  year = {2024},
}
""",
        prompt={
            "query": "Gegeven een titel, haal de wetenschappelijke abstract op die het beste bij de titel past"
        },
    )

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TRECCOVID(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TRECCOVID",
        description="TRECCOVID is an ad-hoc search challenge based on the COVID-19 dataset containing scientific articles related to the COVID-19 pandemic.",
        reference="https://ir.nist.gov/covidSubmit/index.html",
        dataset={
            "path": "mteb/trec-covid",
            "revision": "bb9466bac8153a0349341eb1b22e06409e78ef4e",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2021-12-31"),
        domains=["Medical", "Academic", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{roberts2021searching,
  author = {Roberts, Kirk and Alam, Tasmeer and Bedrick, Steven and Demner-Fushman, Dina and Lo, Kyle and Soboroff, Ian and Voorhees, Ellen and Wang, Lucy Lu and Hersh, William R.},
  doi = {10.1016/j.jbi.2021.103865},
  issn = {1532-0464},
  journal = {Journal of Biomedical Informatics},
  month = {Sept},
  pages = {103865},
  publisher = {Elsevier BV},
  title = {Searching for scientific evidence in a pandemic: An overview of TREC-COVID},
  url = {https://doi.org/10.1016/j.jbi.2021.103865},
  volume = {121},
  year = {2021},
}
""",
        prompt={
            "query": "Given a query on COVID-19, retrieve documents that answer the query"
        },
    )

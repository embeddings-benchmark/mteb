from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


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
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Medical", "Academic", "Written"],
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@misc{roberts2021searching,
  archiveprefix = {arXiv},
  author = {Kirk Roberts and Tasmeer Alam and Steven Bedrick and Dina Demner-Fushman and Kyle Lo and Ian Soboroff and Ellen Voorhees and Lucy Lu Wang and William R Hersh},
  eprint = {2104.09632},
  primaryclass = {cs.IR},
  title = {Searching for Scientific Evidence in a Pandemic: An Overview of TREC-COVID},
  year = {2021},
}
""",
        prompt={
            "query": "Given a query on COVID-19, retrieve documents that answer the query"
        },
    )

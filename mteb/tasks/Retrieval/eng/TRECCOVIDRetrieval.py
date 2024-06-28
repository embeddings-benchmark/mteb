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
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@misc{roberts2021searching,
      title={Searching for Scientific Evidence in a Pandemic: An Overview of TREC-COVID}, 
      author={Kirk Roberts and Tasmeer Alam and Steven Bedrick and Dina Demner-Fushman and Kyle Lo and Ian Soboroff and Ellen Voorhees and Lucy Lu Wang and William R Hersh},
      year={2021},
      eprint={2104.09632},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        n_samples=None,
        avg_character_length={
            "test": {
                "average_document_length": 1116.7434221277986,
                "average_query_length": 69.24,
                "num_documents": 171332,
                "num_queries": 50,
                "average_relevant_docs_per_query": 493.5,
            }
        },
    )

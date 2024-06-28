from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class RiaNewsRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RiaNewsRetrieval",
        dataset={
            "path": "ai-forever/ria-news-retrieval",
            "revision": "82374b0bbacda6114f39ff9c5b925fa1512ca5d7",
        },
        description="News article retrieval by headline. Based on Rossiya Segodnya dataset.",
        reference="https://arxiv.org/abs/1901.07786",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="ndcg_at_10",
        date=("2010-01-01", "2014-12-31"),
        form=["written"],
        domains=["News"],
        task_subtypes=["Article retrieval"],
        license="cc-by-nc-nd-4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{gavrilov2018self,
        title={Self-Attentive Model for Headline Generation},
        author={Gavrilov, Daniil and  Kalaidin, Pavel and  Malykh, Valentin},
        booktitle={Proceedings of the 41st European Conference on Information Retrieval},
        year={2019}
        }""",
        n_samples={"test": 10000},
        avg_character_length={
            "test": {
                "average_document_length": 1165.6429557148213,
                "average_query_length": 62.4029,
                "num_documents": 704344,
                "num_queries": 10000,
                "average_relevant_docs_per_query": 1.0,
            }
        },
    )

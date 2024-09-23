from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class RiaNewsRetrievalFast(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RiaNewsRetrieval-Fast",
        dataset={
            "path": f"mteb/RiaNewsRetrieval_test_top_250_only_w_correct",
            "revision": "latest",
        },
        description="News article retrieval by headline. Based on Rossiya Segodnya dataset.",
        reference="https://arxiv.org/abs/1901.07786",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="ndcg_at_10",
        date=("2010-01-01", "2014-12-31"),
        domains=["News", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-nc-nd-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{gavrilov2018self,
        title={Self-Attentive Model for Headline Generation},
        author={Gavrilov, Daniil and  Kalaidin, Pavel and  Malykh, Valentin},
        booktitle={Proceedings of the 41st European Conference on Information Retrieval},
        year={2019}
        }""",
        descriptive_stats={
            "n_samples": {"test": 10000},
            "avg_character_length": {
                "test": {"average_document_length": 1194.344323696284, "average_query_length": 62.57272727272727, "num_documents": 604196, "num_queries": 990, "average_relevant_docs_per_query": 1.0}
            },
        },
    )

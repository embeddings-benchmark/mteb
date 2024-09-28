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
                "test": {
                    "average_document_length": 1165.6429557148213,
                    "average_query_length": 62.4029,
                    "num_documents": 704344,
                    "num_queries": 10000,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )


class RiaNewsRetrievalHardNegatives(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="RiaNewsRetrievalHardNegatives",
        dataset={
            "path": "mteb/RiaNewsRetrieval_test_top_250_only_w_correct-v2",
            "revision": "d42860a6c15f0a2c4485bda10c6e5b641fdfe479",
        },
        description="News article retrieval by headline. Based on Rossiya Segodnya dataset. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
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
            "n_samples": {"test": 1000},
            "avg_character_length": {
                "test": {
                    "average_document_length": 1225.7253146619116,
                    "average_query_length": 62.338,
                    "num_documents": 191237,
                    "num_queries": 1000,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

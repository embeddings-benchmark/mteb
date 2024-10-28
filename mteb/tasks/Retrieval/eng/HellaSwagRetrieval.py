from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class HellaSwag(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HellaSwag",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on HellaSwag.",
        reference="https://rowanzellers.com/hellaswag/",
        dataset={
            "path": "RAR-b/hellaswag",
            "revision": "a5c990205e017d10761197ccab3000936689c3ae",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{xiao2024rar,
  title={RAR-b: Reasoning as Retrieval Benchmark},
  author={Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2404.06347},
  year={2024}
}
@article{zellers2019hellaswag,
  title={Hellaswag: Can a machine really finish your sentence?},
  author={Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
  journal={arXiv preprint arXiv:1905.07830},
  year={2019}
}
""",
        descriptive_stats={
            "n_samples": {"test": 10042},
            "avg_character_length": {
                "test": {
                    "average_document_length": 137.36519014671472,
                    "average_query_length": 224.53654650468033,
                    "num_documents": 199162,
                    "num_queries": 10042,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

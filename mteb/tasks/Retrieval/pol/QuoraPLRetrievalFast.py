from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class QuoraPLRetrievalFast(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Quora-PL-Fast",
        description="QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.",
        reference="https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
        dataset={
            "path": "mteb/quora-pl_test_top_250_only_w_correct",
            "revision": "main",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=""""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        descriptive_stats={
            "n_samples": {"test": 1000},
            "avg_character_length": {
                "validation": {
                    "average_document_length": 65.82473022253414,
                    "average_query_length": 54.6006,
                    "num_documents": 522931,
                    "num_queries": 5000,
                    "average_relevant_docs_per_query": 1.5252,
                },
                "test": {
                    "average_document_length": 65.82473022253414,
                    "average_query_length": 54.5354,
                    "num_documents": 522931,
                    "num_queries": 10000,
                    "average_relevant_docs_per_query": 1.5675,
                },
            },
        },
    )

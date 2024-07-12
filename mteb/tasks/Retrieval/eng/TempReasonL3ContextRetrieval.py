from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class TempReasonL3Context(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TempReasonL3Context",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l3-context.",
        reference="https://github.com/DAMO-NLP-SG/TempReason",
        dataset={
            "path": "RAR-b/TempReason-l3-context",
            "revision": "3c42539652de3d787cecfb897d3b20905e5c7250",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="CC BY-SA 3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{xiao2024rar,
  title={RAR-b: Reasoning as Retrieval Benchmark},
  author={Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2404.06347},
  year={2024}
}
@article{tan2023towards,
  title={Towards benchmarking and improving the temporal reasoning capability of large language models},
  author={Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
  journal={arXiv preprint arXiv:2306.08952},
  year={2023}
}
""",
        descriptive_stats={
            "n_samples": {"test": 4426},
            "avg_character_length": {
                "test": {
                    "average_document_length": 19.80534984678243,
                    "average_query_length": 13424.633077270673,
                    "num_documents": 15664,
                    "num_queries": 4426,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

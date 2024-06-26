from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class TempReasonL3Pure(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TempReasonL3Pure",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l3-pure.",
        reference="https://github.com/DAMO-NLP-SG/TempReason",
        dataset={
            "path": "RAR-b/TempReason-l3-pure",
            "revision": "68fba138e7e63daccecfbdad0a9d2714e56e34ff",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Reasoning as Retrieval"],
        license="CC BY-SA 3.0",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
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
        n_samples={"test": 4426},
        avg_character_length={
            "test": {
                "average_document_length": 19.80534984678243,
                "average_query_length": 74.44012652507908,
                "num_documents": 15664,
                "num_queries": 4426,
                "average_relevant_docs_per_query": 1.0,
            }
        },
    )

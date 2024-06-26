from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SIQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SIQA",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on SIQA.",
        reference="https://leaderboard.allenai.org/socialiqa/submissions/get-started",
        dataset={
            "path": "RAR-b/siqa",
            "revision": "4ed8415e9dc24060deefc84be59e2db0aacbadcc",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Reasoning as Retrieval"],
        license="CC BY",
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
@article{sap2019socialiqa,
  title={Socialiqa: Commonsense reasoning about social interactions},
  author={Sap, Maarten and Rashkin, Hannah and Chen, Derek and LeBras, Ronan and Choi, Yejin},
  journal={arXiv preprint arXiv:1904.09728},
  year={2019}
}
""",
        n_samples={"test": 0},
        avg_character_length={
            "test": {
                "average_document_length": 22.967085695044617,
                "average_query_length": 127.75383828045035,
                "num_documents": 71276,
                "num_queries": 1954,
                "average_relevant_docs_per_query": 1.0,
            }
        },
    )

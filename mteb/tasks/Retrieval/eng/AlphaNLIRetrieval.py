from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class AlphaNLI(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AlphaNLI",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on AlphaNLI.",
        reference="https://leaderboard.allenai.org/anli/submissions/get-started",
        dataset={
            "path": "RAR-b/alphanli",
            "revision": "303f40ef3d50918d3dc43577d33f2f7344ad72c1",
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
        license="CC BY-NC 4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{xiao2024rar,
  title={RAR-b: Reasoning as Retrieval Benchmark},
  author={Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2404.06347},
  year={2024}
}

@article{bhagavatula2019abductive,
  title={Abductive commonsense reasoning},
  author={Bhagavatula, Chandra and Bras, Ronan Le and Malaviya, Chaitanya and Sakaguchi, Keisuke and Holtzman, Ari and Rashkin, Hannah and Downey, Doug and Yih, Scott Wen-tau and Choi, Yejin},
  journal={arXiv preprint arXiv:1908.05739},
  year={2019}
}
""",
        descriptive_stats={
            "n_samples": {"test": 1532},
            "avg_character_length": {
                "test": {
                    "average_document_length": 43.42647308646886,
                    "average_query_length": 103.05483028720627,
                    "num_documents": 241347,
                    "num_queries": 1532,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

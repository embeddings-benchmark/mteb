from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class Quail(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Quail",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on Quail.",
        reference="https://text-machine.cs.uml.edu/lab2/projects/quail/",
        dataset={
            "path": "RAR-b/quail",
            "revision": "1851bc536f8bdab29e03e29191c4586b1d8d7c5a",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="CC BY-NC-SA 4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{xiao2024rar,
  title={RAR-b: Reasoning as Retrieval Benchmark},
  author={Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2404.06347},
  year={2024}
}
@inproceedings{rogers2020getting,
  title={Getting closer to AI complete question answering: A set of prerequisite real tasks},
  author={Rogers, Anna and Kovaleva, Olga and Downey, Matthew and Rumshisky, Anna},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={34},
  number={05},
  pages={8722--8731},
  year={2020}
}
""",
        descriptive_stats={
            "n_samples": {"test": 2720},
            "avg_character_length": {
                "test": {
                    "average_document_length": 27.50788422240522,
                    "average_query_length": 1957.3632352941177,
                    "num_documents": 32787,
                    "num_queries": 2720,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class WinoGrande(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WinoGrande",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on winogrande.",
        reference="https://winogrande.allenai.org/",
        dataset={
            "path": "RAR-b/winogrande",
            "revision": "f74c094f321077cf909ddfb8bccc1b5912a4ac28",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="CC BY",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{xiao2024rar,
  title={RAR-b: Reasoning as Retrieval Benchmark},
  author={Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2404.06347},
  year={2024}
}
@article{sakaguchi2021winogrande,
  title={Winogrande: An adversarial winograd schema challenge at scale},
  author={Sakaguchi, Keisuke and Bras, Ronan Le and Bhagavatula, Chandra and Choi, Yejin},
  journal={Communications of the ACM},
  volume={64},
  number={9},
  pages={99--106},
  year={2021},
  publisher={ACM New York, NY, USA}
}
""",
        descriptive_stats={
            "n_samples": {"test": 0},
            "avg_character_length": {
                "test": {
                    "average_document_length": 7.68243375858685,
                    "average_query_length": 111.78216258879242,
                    "num_documents": 5095,
                    "num_queries": 1267,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class PIQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PIQA",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on PIQA.",
        reference="https://arxiv.org/abs/1911.11641",
        dataset={
            "path": "RAR-b/piqa",
            "revision": "bb30be7e9184e6b6b1d99bbfe1bb90a3a81842e6",
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
        license="AFL-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{xiao2024rar,
  title={RAR-b: Reasoning as Retrieval Benchmark},
  author={Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2404.06347},
  year={2024}
}
@inproceedings{bisk2020piqa,
  title={Piqa: Reasoning about physical commonsense in natural language},
  author={Bisk, Yonatan and Zellers, Rowan and Gao, Jianfeng and Choi, Yejin and others},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={34},
  number={05},
  pages={7432--7439},
  year={2020}
}
""",
        descriptive_stats={
            "n_samples": {"test": 1838},
            "avg_character_length": {
                "test": {
                    "average_document_length": 99.89012998705756,
                    "average_query_length": 36.08052230685528,
                    "num_documents": 35542,
                    "num_queries": 1838,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

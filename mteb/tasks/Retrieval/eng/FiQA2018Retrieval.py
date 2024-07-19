from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class FiQA2018(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="FiQA2018",
        description="Financial Opinion Mining and Question Answering",
        reference="https://sites.google.com/view/fiqa/",
        dataset={
            "path": "mteb/fiqa",
            "revision": "27a168819829fe9bcd655c2df245fb19452e8e06",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train", "dev", "test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@inproceedings{
thakur2021beir,
title={{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
author={Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
year={2021},
url={https://openreview.net/forum?id=wCu6T5xFjeJ}
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "train": {
                    "average_document_length": 767.2108157812554,
                    "average_query_length": 61.49763636363636,
                    "num_documents": 57638,
                    "num_queries": 5500,
                    "average_relevant_docs_per_query": 2.5756363636363635,
                },
                "dev": {
                    "average_document_length": 767.2108157812554,
                    "average_query_length": 62.756,
                    "num_documents": 57638,
                    "num_queries": 500,
                    "average_relevant_docs_per_query": 2.476,
                },
                "test": {
                    "average_document_length": 767.2108157812554,
                    "average_query_length": 62.7037037037037,
                    "num_documents": 57638,
                    "num_queries": 648,
                    "average_relevant_docs_per_query": 2.632716049382716,
                },
            },
        },
    )

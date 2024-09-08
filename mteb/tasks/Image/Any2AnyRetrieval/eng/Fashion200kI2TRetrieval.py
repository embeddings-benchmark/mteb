from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class Fashion200kI2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="Fashion200kI2TRetrieval",
        description="Retrieve clothes based on descriptions.",
        reference="https://openaccess.thecvf.com/content_iccv_2017/html/Han_Automatic_Spatially-Aware_Fashion_ICCV_2017_paper.html",
        dataset={
            "path": "MRBench/mbeir_fashion200k_task3",
            "revision": "96a313715ecf67f5dfe70c4fa52406bc7bdfbeee",
            # "trust_remote_code": True,
        },
        type="Retrieval",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2017-01-01", "2017-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="Apache-2.0",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{han2017automatic,
  title={Automatic spatially-aware fashion concept discovery},
  author={Han, Xintong and Wu, Zuxuan and Huang, Phoenix X and Zhang, Xiao and Zhu, Menglong and Li, Yuan and Zhao, Yang and Davis, Larry S},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={1463--1471},
  year={2017}
}""",
        descriptive_stats={
            "n_samples": {"test": 4890},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 61700,
                    "num_queries": 4890,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

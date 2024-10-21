from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class Fashion200kT2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="Fashion200kT2IRetrieval",
        description="Retrieve clothes based on descriptions.",
        reference="https://openaccess.thecvf.com/content_iccv_2017/html/Han_Automatic_Spatially-Aware_Fashion_ICCV_2017_paper.html",
        dataset={
            "path": "MRBench/mbeir_fashion200k_task0",
            "revision": "1b86e2dde50e671d5c83d07a79e8b1d8c696964b",
            # "trust_remote_code": True,
        },
        type="Retrieval",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2017-01-01", "2017-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="Apache-2.0",
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
            "n_samples": {"test": 1719},
            "avg_character_length": {
                "test": {
                    "average_document_length": 30.94235294117647,
                    "average_query_length": 131.56569965870307,
                    "num_documents": 201824,
                    "num_queries": 1719,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

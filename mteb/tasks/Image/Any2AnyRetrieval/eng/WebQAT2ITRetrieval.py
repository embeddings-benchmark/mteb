from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class WebQAT2ITRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="WebQAT2ITRetrieval",
        description="Retrieve sources of information based on questions.",
        reference="https://openaccess.thecvf.com/content/CVPR2022/html/Chang_WebQA_Multihop_and_Multimodal_QA_CVPR_2022_paper.html",
        dataset={
            "path": "MRBench/mbeir_webqa_task2",
            "revision": "53db4c9f9c93cb74926a1c9d04dea7d7acac2f21",
        },
        type="Any2AnyRetrieval",
        category="t2it",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{chang2022webqa,
  author = {Chang, Yingshan and Narang, Mridu and Suzuki, Hisami and Cao, Guihong and Gao, Jianfeng and Bisk, Yonatan},
  booktitle = {Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages = {16495--16504},
  title = {Webqa: Multihop and multimodal qa},
  year = {2022},
}
""",
        prompt={"query": "Find a Wikipedia image that answers this question."},
        descriptive_stats={
            "n_samples": {"test": 2511},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 403196,
                    "num_queries": 2511,
                    "average_relevant_docs_per_query": 1.4,
                }
            },
        },
    )

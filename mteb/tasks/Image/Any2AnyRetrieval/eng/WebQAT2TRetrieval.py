from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class WebQAT2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="WebQAT2TRetrieval",
        description="Retrieve sources of information based on questions.",
        reference="https://openaccess.thecvf.com/content/CVPR2022/html/Chang_WebQA_Multihop_and_Multimodal_QA_CVPR_2022_paper.html",
        dataset={
            "path": "MRBench/mbeir_webqa_task1",
            "revision": "468b42a2b2e767d80d2d93f5ae5d42f135a10478",
        },
        type="Any2AnyRetrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text"],
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
        prompt={
            "query": "Retrieve passages from Wikipedia that provide answers to the following question."
        },
        descriptive_stats={
            "n_samples": {"test": 2455},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 544457,
                    "num_queries": 2455,
                    "average_relevant_docs_per_query": 2.0,
                }
            },
        },
    )

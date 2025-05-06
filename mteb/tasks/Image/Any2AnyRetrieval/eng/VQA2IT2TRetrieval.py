from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class VQA2IT2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VQA2IT2TRetrieval",
        description="Retrieve the correct answer for a question about an image.",
        reference="https://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html",
        dataset={
            "path": "JamieSJS/vqa-2",
            "revision": "69882b6ba0b443dd62e633e546725b0f13b7e3aa",
            "trust_remote_code": True,
        },
        type="Any2AnyRetrieval",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2017-07-01", "2017-07-01"),
        domains=["Web"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Goyal_2017_CVPR,
  author = {Goyal, Yash and Khot, Tejas and Summers-Stay, Douglas and Batra, Dhruv and Parikh, Devi},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {July},
  title = {Making the v in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering},
  year = {2017},
}
""",
        descriptive_stats={
            "n_samples": {"test": 4319},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 2091,
                    "num_queries": 4319,
                    "average_relevant_docs_per_query": 1,
                }
            },
        },
    )

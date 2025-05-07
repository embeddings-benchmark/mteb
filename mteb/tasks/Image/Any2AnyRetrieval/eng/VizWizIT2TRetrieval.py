from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class VizWizIT2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VizWizIT2TRetrieval",
        description="Retrieve the correct answer for a question about an image.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/papers/Gurari_VizWiz_Grand_Challenge_CVPR_2018_paper.pdf",
        dataset={
            "path": "JamieSJS/vizwiz",
            "revision": "044af162d55f82ab603fa16ffcf7f1e4dbf300e9",
            "trust_remote_code": True,
        },
        type="Any2AnyRetrieval",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-01-01"),
        domains=["Web"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{gurari2018vizwiz,
  author = {Gurari, Danna and Li, Qing and Stangl, Abigale J and Guo, Anhong and Lin, Chi and Grauman, Kristen and Luo, Jiebo and Bigham, Jeffrey P},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages = {3608--3617},
  title = {Vizwiz grand challenge: Answering visual questions from blind people},
  year = {2018},
}
""",
        descriptive_stats={
            "n_samples": {"test": 214354},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 2143540,
                    "num_queries": 214354,
                    "average_relevant_docs_per_query": 1,
                }
            },
        },
    )

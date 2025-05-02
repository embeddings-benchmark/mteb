from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class OVENIT2ITRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="OVENIT2ITRetrieval",
        description="Retrieval a Wiki image and passage to answer query about an image.",
        reference="https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Open-domain_Visual_Entity_Recognition_Towards_Recognizing_Millions_of_Wikipedia_Entities_ICCV_2023_paper.html",
        dataset={
            "path": "MRBench/mbeir_oven_task8",
            "revision": "350d14b7258189654e26a2be93dc0bd6bee09b76",
        },
        type="Any2AnyRetrieval",
        category="it2it",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{hu2023open,
  author = {Hu, Hexiang and Luan, Yi and Chen, Yang and Khandelwal, Urvashi and Joshi, Mandar and Lee, Kenton and Toutanova, Kristina and Chang, Ming-Wei},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages = {12065--12075},
  title = {Open-domain visual entity recognition: Towards recognizing millions of wikipedia entities},
  year = {2023},
}
""",
        prompt={
            "query": "Retrieve a Wikipedia image-description pair that provides evidence for the question of this image."
        },
        descriptive_stats={
            "n_samples": {"test": 14741},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 335135,
                    "num_queries": 14741,
                    "average_relevant_docs_per_query": 17.7,
                }
            },
        },
    )

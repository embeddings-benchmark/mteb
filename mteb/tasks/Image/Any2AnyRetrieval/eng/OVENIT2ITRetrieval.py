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
        type="Retrieval",
        category="it2it",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="CC BY-SA 4.0",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{hu2023open,
  title={Open-domain visual entity recognition: Towards recognizing millions of wikipedia entities},
  author={Hu, Hexiang and Luan, Yi and Chen, Yang and Khandelwal, Urvashi and Joshi, Mandar and Lee, Kenton and Toutanova, Kristina and Chang, Ming-Wei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12065--12075},
  year={2023}
}""",
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

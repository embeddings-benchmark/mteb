from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class FashionIQIT2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="FashionIQIT2IRetrieval",
        description="Retrieve clothes based on descriptions.",
        reference="https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Fashion_IQ_A_New_Dataset_Towards_Retrieving_Images_by_Natural_CVPR_2021_paper.html",
        dataset={
            "path": "MRBench/mbeir_fashioniq_task7",
            "revision": "e6f0ec70becc413d940cd62b2cfa3b1d3a08c31a",
            # "trust_remote_code": True,
        },
        type="Any2AnyRetrieval",
        category="it2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation="""@inproceedings{wu2021fashion,
  title={Fashion iq: A new dataset towards retrieving images by natural language feedback},
  author={Wu, Hui and Gao, Yupeng and Guo, Xiaoxiao and Al-Halah, Ziad and Rennie, Steven and Grauman, Kristen and Feris, Rogerio},
  booktitle={Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition},
  pages={11307--11317},
  year={2021}
}""",
        prompt={
            "query": "Find a fashion image that aligns with the reference image and style note."
        },
        descriptive_stats={
            "n_samples": {"test": 6003},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 74381,
                    "num_queries": 6003,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class AutoRAGRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AutoRAGRetrieval",
        description="AutoRAGRetrieval",
        reference=None,
        dataset={
            "path": "nlpai-lab/markers_bm",
            "revision": "e252a2b2644140a68961e64f10c107e5c036119b",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@misc{AutoRAG,
  author = "Dongkyu Kim and Byoungwook Kim and Donggeon Han",
  title = "AutoRAG",
  year = {2024},
  url = {https://github.com/Marker-Inc-Korea/AutoRAG}
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "dev": {
                    "average_document_length": 305.15457788347203,
                    "average_query_length": 22.62442640310625,
                    "num_documents": 9251,
                    "num_queries": 2833,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

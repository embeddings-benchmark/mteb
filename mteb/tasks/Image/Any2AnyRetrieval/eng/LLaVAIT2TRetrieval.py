from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class LLaVAIT2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="LLaVAIT2TRetrieval",
        description="Retrieve responses to answer questions about images.",
        reference="https://github.com/LinWeizheDragon/FLMR/blob/main/docs/Datasets.md",
        dataset={
            "path": "izhx/UMRB-LLaVA",
            "revision": "2a5ed414aab388d8cdd244ba2cf8c8960df4d44d",
        },
        type="Any2AnyRetrieval",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2024-07-06", "2024-02-26"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{lin-etal-2024-preflmr,
    title = "{P}re{FLMR}: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers",
    author = "Lin, Weizhe  and
      Mei, Jingbiao  and
      Chen, Jinghong  and
      Byrne, Bill",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.289",
    doi = "10.18653/v1/2024.acl-long.289",
    pages = "5294--5316",
}""",
        prompt={
            "query": "Provide a specific decription of the image along with the following question."
        },
        descriptive_stats={
            "n_samples": {"test": 5120},
            "avg_character_length": {
                "test": {
                    "average_document_length": 546.1925258591925,
                    "average_query_length": 59.580859375,
                    "num_documents": 5994,
                    "num_queries": 5120,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

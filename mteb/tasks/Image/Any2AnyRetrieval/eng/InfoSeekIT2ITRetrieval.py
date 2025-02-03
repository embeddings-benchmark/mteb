from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class InfoSeekIT2ITRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="InfoSeekIT2ITRetrieval",
        description="Retrieve source text and image information to answer questions about images.",
        reference="https://aclanthology.org/2023.emnlp-main.925",
        dataset={
            "path": "MRBench/mbeir_infoseek_task8",
            "revision": "78ee7f7708aac75d3afac5dcab1c9e03cb62664c",
            "trust_remote_code": True,
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
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{chen2023can,
  title={Can Pre-trained Vision and Language Models Answer Visual Information-Seeking Questions?},
  author={Chen, Yang and Hu, Hexiang and Luan, Yi and Sun, Haitian and Changpinyo, Soravit and Ritter, Alan and Chang, Ming-Wei},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={14948--14968},
  year={2023}
}""",
        prompt={
            "query": "Find an image and subject description from Wikipedia that answers my question about this image."
        },
        descriptive_stats={
            "n_samples": {"test": 17593},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 481782,
                    "num_queries": 17593,
                    "average_relevant_docs_per_query": 7.5,
                }
            },
        },
    )

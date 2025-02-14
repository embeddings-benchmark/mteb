from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class InfoSeekIT2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="InfoSeekIT2TRetrieval",
        description="Retrieve source information to answer questions about images.",
        reference="https://aclanthology.org/2023.emnlp-main.925",
        dataset={
            "path": "MRBench/mbeir_infoseek_task6",
            "revision": "d4f4606f7a42bbf311c2957419ef3734fe81c47f",
            "trust_remote_code": True,
        },
        type="Any2AnyRetrieval",
        category="it2t",
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
            "query": "Find a paragraph from Wikipedia that answers my question about this image."
        },
        descriptive_stats={
            "n_samples": {"test": 11323},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 611651,
                    "num_queries": 11323,
                    "average_relevant_docs_per_query": 6.5,
                }
            },
        },
    )

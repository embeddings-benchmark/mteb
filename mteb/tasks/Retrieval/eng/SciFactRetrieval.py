from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SciFact(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciFact",
        dataset={
            "path": "mteb/scifact",
            "revision": "0228b52cf27578f30900b9e5271d331663a030d7",
        },
        description="SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
        reference="https://github.com/allenai/scifact",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train", "test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@inproceedings{specter2020cohan,
  title={SPECTER: Document-level Representation Learning using Citation-informed Transformers},
  author={Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
  booktitle={ACL},
  year={2020}
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "train": {
                    "average_document_length": 1498.4152035500674,
                    "average_query_length": 88.58838071693448,
                    "num_documents": 5183,
                    "num_queries": 809,
                    "average_relevant_docs_per_query": 1.1359703337453646,
                },
                "test": {
                    "average_document_length": 1498.4152035500674,
                    "average_query_length": 90.34666666666666,
                    "num_documents": 5183,
                    "num_queries": 300,
                    "average_relevant_docs_per_query": 1.13,
                },
            },
        },
    )

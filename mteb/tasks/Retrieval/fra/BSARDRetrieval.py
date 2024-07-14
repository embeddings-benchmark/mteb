from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class BSARDRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="BSARDRetrieval",
        description="The Belgian Statutory Article Retrieval Dataset (BSARD) is a French native dataset for studying legal information retrieval. BSARD consists of more than 22,600 statutory articles from Belgian law and about 1,100 legal questions posed by Belgian citizens and labeled by experienced jurists with relevant articles from the corpus.",
        reference="https://huggingface.co/datasets/maastrichtlawtech/bsard",
        dataset={
            "path": "maastrichtlawtech/bsard",
            "revision": "5effa1b9b5fa3b0f9e12523e6e43e5f86a6e6d59",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="recall_at_100",
        date=("2021-05-01", "2021-08-26"),
        domains=["Legal", "Spoken"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{louis2022statutory,
  title = {A Statutory Article Retrieval Dataset in French},
  author = {Louis, Antoine and Spanakis, Gerasimos},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  month = may,
  year = {2022},
  address = {Dublin, Ireland},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2022.acl-long.468/},
  doi = {10.18653/v1/2022.acl-long.468},
  pages = {6789–6803},
}""",
        descriptive_stats={
            "n_samples": {"test": 222},
            "avg_character_length": {
                "test": {
                    "average_document_length": 880.2900631820793,
                    "average_query_length": 144.77027027027026,
                    "num_documents": 22633,
                    "num_queries": 222,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset, only test split
        corpus_raw = datasets.load_dataset(
            name="corpus",
            split="corpus",
            **self.metadata_dict["dataset"],
        )
        queries_raw = datasets.load_dataset(
            name="questions",
            split=self.metadata.eval_splits[0],
            **self.metadata_dict["dataset"],
        )

        self.queries = {
            self.metadata.eval_splits[0]: {
                str(q["id"]): (q["question"] + " " + q["extra_description"]).strip()
                for q in queries_raw
            }
        }

        self.corpus = {
            self.metadata.eval_splits[0]: {
                str(d["id"]): {"text": d["article"]} for d in corpus_raw
            }
        }

        self.relevant_docs = {self.metadata.eval_splits[0]: {}}
        for q in queries_raw:
            for doc_id in q["article_ids"]:
                self.relevant_docs[self.metadata.eval_splits[0]][str(q["id"])] = {
                    str(doc_id): 1
                }

        self.data_loaded = True

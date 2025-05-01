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
        bibtex_citation=r"""
@inproceedings{louis2022statutory,
  address = {Dublin, Ireland},
  author = {Louis, Antoine and Spanakis, Gerasimos},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  doi = {10.18653/v1/2022.acl-long.468},
  month = may,
  pages = {6789–6803},
  publisher = {Association for Computational Linguistics},
  title = {A Statutory Article Retrieval Dataset in French},
  url = {https://aclanthology.org/2022.acl-long.468/},
  year = {2022},
}
""",
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

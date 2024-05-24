from __future__ import annotations

import hashlib

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

_EVAL_SPLIT = "test"


class GermanGovServiceRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GermanGovServiceRetrieval",
        description="LHM-Dienstleistungen-QA is a German question answering dataset for government services of the Munich city administration. It associates questions with a textual context containing the answer",
        reference="https://huggingface.co/datasets/it-at-m/LHM-Dienstleistungen-QA",
        dataset={
            "path": "it-at-m/LHM-Dienstleistungen-QA",
            "revision": "ed40131b56ce86ce3666f2942953595dd9d29608",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_5",
        date=("2022-11-01", "2022-11-30"),
        form=["written"],
        domains=["Government"],
        task_subtypes=["Question answering"],
        license="mit",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=[],
        bibtex_citation="""@software{lhm-dienstleistungen-qa,
  author       = {Schröder, Leon Marius and
                  Gutknecht, Clemens and
                  Alkiddeh, Oubada and 
                  Susanne Weiß,
                  Lukas, Leon},
  title        = {LHM-Dienstleistungen-QA - german public domain question-answering dataset},
  month        = nov,
  year         = 2022,
  publisher    = {it@M},
  url          = {https://huggingface.co/datasets/it-at-m/LHM-Dienstleistungen-QA}
}""",
        text_creation="found",
        n_samples={"test": 357},
        avg_character_length={"test": 1211.69},
    )

    @staticmethod
    def get_hash(input_str) -> str:
        return hashlib.md5(input_str.encode("utf-8")).hexdigest()

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        dataset = datasets.load_dataset(
            path=self.metadata_dict["dataset"]["path"],
            split=_EVAL_SPLIT,
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )
        corpus = {}
        queries = {}
        relevant_docs = {}

        for row in dataset:  # row: title, context, question, ...
            # use hash values as IDs
            d_id = "d_" + self.get_hash(row["title"] + row["context"])
            q_id = "q_" + self.get_hash(row["question"])

            corpus[d_id] = {
                "_id": d_id,
                "title": row["title"],
                "text": row["context"],
            }
            queries[q_id] = row["question"]

            if q_id not in relevant_docs:
                relevant_docs[q_id] = {}

            relevant_docs[q_id][d_id] = 1  # 1 = relevant

        self.queries = {_EVAL_SPLIT: queries}
        self.corpus = {_EVAL_SPLIT: corpus}
        self.relevant_docs = {_EVAL_SPLIT: relevant_docs}

        self.data_loaded = True

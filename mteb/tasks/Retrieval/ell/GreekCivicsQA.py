from __future__ import annotations

from hashlib import sha256

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class GreekCivicsQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GreekCivicsQA",
        description="This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school",
        reference="https://huggingface.co/datasets/antoinelb7/alloprof",
        dataset={
            "path": "ilsp/greek_civics_qa",
            "revision": "a04523a3c83153be07a8945bb1fb351cbbcef90b",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["default"],
        eval_langs=["ell-Grek"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2024-04-01"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"default": 407},
        avg_character_length={"default": 2226.85},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset
        eval_split = self.metadata_dict["eval_splits"][0]
        data_raw = datasets.load_dataset(**self.metadata_dict["dataset"])[eval_split]

        queries = {eval_split: {}}
        corpus = {eval_split: {}}
        relevant_docs = {eval_split: {}}

        question_ids = {
            question: str(id) for id, question in zip(data_raw["id"], data_raw["question"])
        }

        context_ids = {
            answer: sha256(answer.encode("utf-8")).hexdigest()
            for answer in set(data_raw["answer"])
        }

        for row in data_raw:
            question = row["question"]
            context = row["answer"]
            query_id = question_ids[question]
            queries[eval_split][query_id] = question

            doc_id = context_ids[context]
            corpus[eval_split][doc_id] = {"text": context}
            if query_id not in relevant_docs[eval_split]:
                relevant_docs[eval_split][query_id] = {}
            relevant_docs[eval_split][query_id][doc_id] = 1

        self.corpus = datasets.DatasetDict(corpus)
        self.queries = datasets.DatasetDict(queries)
        self.relevant_docs = datasets.DatasetDict(relevant_docs)

        self.data_loaded = True

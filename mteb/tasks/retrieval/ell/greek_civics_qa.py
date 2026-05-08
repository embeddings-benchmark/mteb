from hashlib import sha256

import datasets
from datasets import Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="t2t",
        modalities=["text"],
        eval_splits=["default"],
        eval_langs=["ell-Grek"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2024-04-01"),
        domains=["Academic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return
        # fetch both subsets of the dataset
        eval_split = self.metadata.eval_splits[0]
        data_raw = datasets.load_dataset(**self.metadata.dataset)[eval_split]

        queries_dict = {}
        corpus_dict = {}
        relevant_docs = {}

        question_ids = {
            question: str(id)
            for id, question in zip(data_raw["id"], data_raw["question"])
        }

        context_ids = {
            answer: sha256(answer.encode("utf-8")).hexdigest()
            for answer in set(data_raw["answer"])
        }

        for row in data_raw:
            question = row["question"]
            context = row["answer"]
            query_id = question_ids[question]
            queries_dict[query_id] = question

            doc_id = context_ids[context]
            corpus_dict[doc_id] = {"text": context, "title": ""}
            if query_id not in relevant_docs:
                relevant_docs[query_id] = {}
            relevant_docs[query_id][doc_id] = 1

        corpus_dataset = Dataset.from_list(
            [
                {"id": k, "text": v["text"], "title": v["title"]}
                for k, v in corpus_dict.items()
            ]
        )
        queries_dataset = Dataset.from_list(
            [{"id": k, "text": v} for k, v in queries_dict.items()]
        )

        self.dataset = {
            "default": {
                eval_split: {
                    "corpus": corpus_dataset,
                    "queries": queries_dataset,
                    "relevant_docs": relevant_docs,
                    "top_ranked": None,
                }
            }
        }

        self.data_loaded = True

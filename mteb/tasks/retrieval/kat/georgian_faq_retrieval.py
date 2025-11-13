from datasets import DatasetDict, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"


class GeorgianFAQRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GeorgianFAQRetrieval",
        dataset={
            "path": "jupyterjazz/georgian-faq",
            "revision": "2436d9bda047a80959b034a572fdda4d00c80d2e",
        },
        description=(
            "Frequently asked questions (FAQs) and answers mined from Georgian websites via Common Crawl."
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kat-Geor"],
        main_score="ndcg_at_10",
        domains=["Web", "Written"],
        sample_creation="created",
        reference="https://huggingface.co/datasets/jupyterjazz/georgian-faq",
        date=("2024-05-02", "2024-05-03"),
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        bibtex_citation="",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        queries = {_EVAL_SPLIT: {}}
        corpus = {_EVAL_SPLIT: {}}
        relevant_docs = {_EVAL_SPLIT: {}}

        data = load_dataset(
            self.metadata.dataset["path"],
            split=_EVAL_SPLIT,
            revision=self.metadata.dataset["revision"],
        )

        question_ids = {}
        answer_ids = {}

        for row in data:
            question = row["question"]
            answer = row["answer"]
            if question not in question_ids:
                question_ids[question] = len(question_ids)
            if answer not in answer_ids:
                answer_ids[answer] = len(answer_ids)

        for row in data:
            question = row["question"]
            answer = row["answer"]
            query_id = f"Q{question_ids[question]}"
            queries[_EVAL_SPLIT][query_id] = question
            doc_id = f"D{answer_ids[answer]}"
            corpus[_EVAL_SPLIT][doc_id] = {"text": answer}
            if query_id not in relevant_docs[_EVAL_SPLIT]:
                relevant_docs[_EVAL_SPLIT][query_id] = {}
            relevant_docs[_EVAL_SPLIT][query_id][doc_id] = 1

        self.corpus = DatasetDict(corpus)
        self.queries = DatasetDict(queries)
        self.relevant_docs = DatasetDict(relevant_docs)
        self.data_loaded = True

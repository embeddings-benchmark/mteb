from __future__ import annotations

from datasets import DatasetDict, load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

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
        category="s2p",
        eval_splits=["test"],
        eval_langs=["kat-Geor"],
        main_score="ndcg_at_10",
        domains=["Web"],
        text_creation="created",
        n_samples={_EVAL_SPLIT: 2566},
        reference="https://huggingface.co/datasets/jupyterjazz/georgian-faq",
        date=("2024-05-02", "2024-05-03"),
        form=["written"],
        task_subtypes=["Question answering"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        bibtex_citation="",
        avg_character_length={_EVAL_SPLIT: 572},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        queries = {_EVAL_SPLIT: {}}
        corpus = {_EVAL_SPLIT: {}}
        relevant_docs = {_EVAL_SPLIT: {}}

        data = load_dataset(
            self.metadata_dict["dataset"]["path"],
            split=_EVAL_SPLIT,
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )
        question_ids = {
            question: _id for _id, question in enumerate(set(data["question"]))
        }
        answer_ids = {answer: _id for _id, answer in enumerate(set(data["answer"]))}

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

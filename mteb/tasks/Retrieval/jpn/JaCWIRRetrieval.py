from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class JaCWIRRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="JaCWIRRetrieval",
        description="""JaCWIR is a small-scale Japanese information retrieval evaluation dataset consisting of
5000 question texts and approximately 500k web page titles and web page introductions or summaries
(meta descriptions, etc.). The question texts are created based on one of the 500k web pages,
and that data is used as a positive example for the question text.""",
        reference="https://huggingface.co/datasets/hotchpotch/JaCWIR",
        dataset={
            "path": "sbintuitions/JMTEB",
            "revision": "b194332dfb8476c7bdd0aaf80e2c4f2a0b4274c2",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2000-01-01", "2024-12-31"),
        domains=["Web", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{yuichi-tateno-2024-jacwir,
  author = {Yuichi Tateno},
  title = {JaCWIR: Japanese Casual Web IR - 日本語情報検索評価のための小規模でカジュアルなWebタイトルと概要のデータセット},
  url = {https://huggingface.co/datasets/hotchpotch/JaCWIR},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_list = datasets.load_dataset(
            name="jacwir-retrieval-query",
            split=_EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )

        queries = {}
        qrels = {}
        for row_id, row in enumerate(query_list):
            queries[str(row_id)] = row["query"]
            # Handle relevant_docs which should be a list
            relevant_docs = row["relevant_docs"]
            if not isinstance(relevant_docs, list):
                relevant_docs = [relevant_docs]
            qrels[str(row_id)] = {str(doc_id): 1 for doc_id in relevant_docs}

        corpus_list = datasets.load_dataset(
            name="jacwir-retrieval-corpus",
            split="corpus",
            **self.metadata_dict["dataset"],
        )

        corpus = {str(row["docid"]): {"text": row["text"]} for row in corpus_list}

        self.corpus = {_EVAL_SPLIT: corpus}
        self.queries = {_EVAL_SPLIT: queries}
        self.relevant_docs = {_EVAL_SPLIT: qrels}

        self.data_loaded = True

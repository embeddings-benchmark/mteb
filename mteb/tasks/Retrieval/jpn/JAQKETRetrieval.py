from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class JAQKETRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JAQKETRetrieval",
        description="JAQKET (AIO Ver. 1.0) dataset has a quiz set and a corpus that consists of Wikipedia passages, each of which is a description of an entity (the title of the Wikipedia page). A quiz question is answered by looking for the most relevant Wikipedia passage to the quiz question text.",
        reference="https://github.com/sbintuitions/JMTEB",
        dataset={
            "path": "sbintuitions/JMTEB",
            "revision": "e4af6c73182bebb41d94cb336846e5a452454ea7",
        },
        type="Retrieval",
        category="s2s",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2000-01-01", "2020-12-31"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=None,
        license="cc-by-sa-4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation="""
        @inproceedings{suzuki2020jaqket,
            title={JAQKET: Constructing a Japanese QA dataset based on quiz questions [JAQKET:クイズを題材にした日本語QAデータセットの構築] (in Japanese)},
            author={Masatoshi Suzuki and Jun Suzuki and Koji Matsuda and Kyosuke Nishida and Naoya Inoue},
            booktitle={NLP 2020},
            year={2020},
            url={https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf}
        }
        """,
        n_samples={_EVAL_SPLIT: 997},
        avg_character_length={_EVAL_SPLIT: 1895.97},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_list = datasets.load_dataset(
            name="jaqket-query",
            split=_EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )

        queries = {}
        qrels = {}
        for row in query_list:
            queries[str(query_list["qid"])] = row["query"]
            qrels[str(query_list["qid"])] = {str(row["relevant_docs"][0]): 1}

        corpus_list = datasets.load_dataset(
            name="jaqket-corpus",
            split="corpus",
            **self.metadata_dict["dataset"],
        )

        corpus = {
            str(row["docid"]): {"title": row["title"], "text": row["text"]}
            for row in corpus_list
        }

        self.corpus = {_EVAL_SPLIT: corpus}
        self.queries = {_EVAL_SPLIT: queries}
        self.relevant_docs = {_EVAL_SPLIT: qrels}

        self.data_loaded = True

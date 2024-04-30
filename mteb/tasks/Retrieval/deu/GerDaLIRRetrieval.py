from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class GerDaLIR(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="GerDaLIR",
        description="GerDaLIR is a legal information retrieval dataset created from the Open Legal Data platform.",
        reference="https://github.com/lavis-nlp/GerDaLIR",
        dataset={
            "path": "jinaai/ger_da_lir",
            "revision": "0bb47f1d73827e96964edb84dfe552f62f4fd5eb",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_rows = datasets.load_dataset(
            name="queries",
            split=self._EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )
        corpus_rows = datasets.load_dataset(
            name="corpus",
            split=self._EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )
        qrels_rows = datasets.load_dataset(
            name="qrels",
            split=self._EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )

        self.queries = {
            self._EVAL_SPLIT: {row["_id"]: row["text"] for row in query_rows}
        }
        self.corpus = {self._EVAL_SPLIT: {row["_id"]: row for row in corpus_rows}}
        self.relevant_docs = {
            self._EVAL_SPLIT: {
                row["_id"]: {v: 1 for v in row["text"].split(" ")} for row in qrels_rows
            }
        }

        self.data_loaded = True

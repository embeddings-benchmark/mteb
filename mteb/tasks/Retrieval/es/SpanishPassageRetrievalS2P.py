from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SpanishPassageRetrievalS2P(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SpanishPassageRetrievalS2P",
        description="Test collection for passage retrieval from health-related Web resources in Spanish.",
        reference="https://mklab.iti.gr/results/spanish-passage-retrieval-dataset/",
        dataset={
            "path": "jinaai/spanish_passage_retrieval",
            "revision": "9cddf2ce5209ade52c2115ccfa00eb22c6d3a837",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["es"],
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

        # BUGFIX: the revision is now used
        query_rows = datasets.load_dataset(
            name="queries",
            split="test",
            trust_remote_code=True,
        )
        corpus_rows = datasets.load_dataset(
            name="corpus.documents",
            split="test",
            trust_remote_code=True,
            **self.metadata_dict["dataset"],
        )
        qrels_rows = datasets.load_dataset(
            name="qrels.s2p",
            split="test",
            trust_remote_code=True,
            **self.metadata_dict["dataset"],
        )

        self.queries = {"test": {row["_id"]: row["text"] for row in query_rows}}
        self.corpus = {"test": {row["_id"]: row for row in corpus_rows}}
        self.relevant_docs = {
            "test": {
                row["_id"]: {v: 1 for v in row["text"].split(" ")} for row in qrels_rows
            }
        }

        self.data_loaded = True

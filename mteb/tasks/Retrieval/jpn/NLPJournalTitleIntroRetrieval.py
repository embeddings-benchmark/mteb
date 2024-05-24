from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class NLPJournalTitleIntroRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NLPJournalTitleIntroRetrieval",
        description="This dataset was created from the Japanese NLP Journal LaTeX Corpus. The titles, abstracts and introductions of the academic papers were shuffled. The goal is to find the corresponding introduction with the given title.",
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
        date=("2000-01-01", "2023-12-31"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=None,
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation=None,
        n_samples={_EVAL_SPLIT: 404},
        avg_character_length={_EVAL_SPLIT: 1040.19},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_list = datasets.load_dataset(
            name="nlp_journal_title_intro-query",
            split=_EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )

        queries = {}
        qrels = {}
        for row_id, row in enumerate(query_list):
            queries[str(row_id)] = row["query"]
            qrels[str(row_id)] = {str(row["relevant_docs"]): 1}

        corpus_list = datasets.load_dataset(
            name="nlp_journal_title_intro-corpus",
            split="corpus",
            **self.metadata_dict["dataset"],
        )

        corpus = {str(row["docid"]): {"text": row["text"]} for row in corpus_list}

        self.corpus = {_EVAL_SPLIT: corpus}
        self.queries = {_EVAL_SPLIT: queries}
        self.relevant_docs = {_EVAL_SPLIT: qrels}

        self.data_loaded = True

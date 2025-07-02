from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class NLPJournalAbsArticleRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="NLPJournalAbsArticleRetrieval",
        description=(
            "This dataset was created from the Japanese NLP Journal LaTeX Corpus. "
            "The titles, abstracts and introductions of the academic papers were shuffled. "
            "The goal is to find the corresponding full article with the given abstract."
        ),
        reference="https://huggingface.co/datasets/sbintuitions/JMTEB",
        dataset={
            "path": "sbintuitions/JMTEB",
            "revision": "b194332dfb8476c7bdd0aaf80e2c4f2a0b4274c2",
            "trust_remote_code": True,
            "dataset_version": "latest",
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("1994-10-10", "2025-06-15"),
        domains=["Academic", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    def __init__(self, dataset_version: str = "latest", **kwargs):
        """Initialize the NLPJournalAbsArticleRetrieval task.

        Args:
            dataset_version: Version of the NLP Journal dataset to use.
                           Options: "v1", "v2", "latest". Default is "latest".
            **kwargs: placeholder
        """
        super().__init__(**kwargs)
        self.dataset_version = dataset_version

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        # Add dataset_version to the dataset loading kwargs
        dataset_kwargs = self.metadata_dict["dataset"].copy()
        dataset_kwargs["dataset_version"] = self.dataset_version

        query_list = datasets.load_dataset(
            name="nlp_journal_abs_article-query",
            split=_EVAL_SPLIT,
            **dataset_kwargs,
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
            name="nlp_journal_abs_article-corpus", split="corpus", **dataset_kwargs
        )

        corpus = {str(row["docid"]): {"text": row["text"]} for row in corpus_list}

        self.corpus = {_EVAL_SPLIT: corpus}
        self.queries = {_EVAL_SPLIT: queries}
        self.relevant_docs = {_EVAL_SPLIT: qrels}

        self.data_loaded = True

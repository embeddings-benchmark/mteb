from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class NLPJournalAbsArticleRetrievalV2(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="NLPJournalAbsArticleRetrieval.V2",
        description=(
            "This dataset was created from the Japanese NLP Journal LaTeX Corpus. "
            "The titles, abstracts and introductions of the academic papers were shuffled. "
            "The goal is to find the corresponding full article with the given abstract. "
            "This is the V2 dataset (last updated 2025-06-15)."
        ),
        reference="https://huggingface.co/datasets/sbintuitions/JMTEB",
        dataset={
            "path": "sbintuitions/JMTEB",
            "revision": "b194332dfb8476c7bdd0aaf80e2c4f2a0b4274c2",
            "trust_remote_code": True,
            "dataset_version": "v2",
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
        adapted_from=["NLPJournalAbsArticleRetrieval"],
        bibtex_citation=r"""
@misc{jmteb,
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
  howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
  title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
  year = {2024},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_list = datasets.load_dataset(
            name="nlp_journal_abs_article-query",
            split=_EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )

        queries = {}
        qrels = {}
        for row_id, row in enumerate(query_list):
            queries[str(row_id)] = row["query"]
            qrels[str(row_id)] = {str(row["relevant_docs"]): 1}

        corpus_list = datasets.load_dataset(
            name="nlp_journal_abs_article-corpus",
            split="corpus",
            **self.metadata_dict["dataset"],
        )

        corpus = {str(row["docid"]): {"text": row["text"]} for row in corpus_list}

        self.corpus = {_EVAL_SPLIT: corpus}
        self.queries = {_EVAL_SPLIT: queries}
        self.relevant_docs = {_EVAL_SPLIT: qrels}

        self.data_loaded = True


class NLPJournalAbsArticleRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="NLPJournalAbsArticleRetrieval",
        description=(
            "This dataset was created from the Japanese NLP Journal LaTeX Corpus. "
            "The titles, abstracts and introductions of the academic papers were shuffled. "
            "The goal is to find the corresponding full article with the given abstract. "
            "This is the V1 dataset (last updated 2020-06-15)."
        ),
        reference="https://huggingface.co/datasets/sbintuitions/JMTEB",
        dataset={
            "path": "sbintuitions/JMTEB",
            "revision": "b194332dfb8476c7bdd0aaf80e2c4f2a0b4274c2",
            "trust_remote_code": True,
            "dataset_version": "v1",
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("1994-10-10", "2020-06-15"),
        domains=["Academic", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{jmteb,
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
  howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
  title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
  year = {2024},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_list = datasets.load_dataset(
            name="nlp_journal_abs_article-query",
            split=_EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )

        queries = {}
        qrels = {}
        for row_id, row in enumerate(query_list):
            queries[str(row_id)] = row["query"]
            qrels[str(row_id)] = {str(row["relevant_docs"]): 1}

        corpus_list = datasets.load_dataset(
            name="nlp_journal_abs_article-corpus",
            split="corpus",
            **self.metadata_dict["dataset"],
        )

        corpus = {str(row["docid"]): {"text": row["text"]} for row in corpus_list}

        self.corpus = {_EVAL_SPLIT: corpus}
        self.queries = {_EVAL_SPLIT: queries}
        self.relevant_docs = {_EVAL_SPLIT: qrels}

        self.data_loaded = True

from __future__ import annotations

import json

import datasets

from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

_EVAL_SPLITS = ["dev", "test"]

_LANGS = {
    # <iso_639_3>-<ISO_15924>
    "english": ["eng-Latn"],
    "french": ["fra-Latn"],
}


def _load_statcan_data(
    path: str, langs: list, splits: str, cache_dir: str = None, revision: str = None
):
    queries = {lang: {split: {} for split in splits} for lang in langs}
    corpus = {lang: {split: {} for split in splits} for lang in langs}
    relevant_docs = {lang: {split: {} for split in splits} for lang in langs}

    for split in splits:
        for lang in langs:
            query_table = datasets.load_dataset(
                path,
                f"queries_{lang}",
                split=split,
                cache_dir=cache_dir,
                revision=revision,
            )
            corpus_table = datasets.load_dataset(
                path,
                "corpus",
                split=lang,
                cache_dir=cache_dir,
                revision=revision,
            )

            for row in query_table:
                query = json.loads(row["query"])
                query_id = row["query_id"]
                doc_id = row["doc_id"]
                queries[lang][split][query_id] = query
                if query_id not in relevant_docs[lang][split]:
                    relevant_docs[lang][split][query_id] = {}
                relevant_docs[lang][split][query_id][doc_id] = 1

            for row in corpus_table:
                doc_id = row["doc_id"]
                doc_content = row["doc"]
                corpus[lang][split][doc_id] = {"text": doc_content}

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class StatcanDialogueDatasetRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="StatcanDialogueDatasetRetrieval",
        description="A Dataset for Retrieving Data Tables through Conversations with Genuine Intents, available in English and French.",
        dataset={
            "path": "McGill-NLP/statcan-dialogue-dataset-retrieval",
            "revision": "7a26938c93e99e0759a1df416896bb72527e2f33",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=_EVAL_SPLITS,
        eval_langs=_LANGS,
        main_score="recall_at_10",
        reference="https://mcgill-nlp.github.io/statcan-dialogue-dataset/",
        date=("2020-01-01", "2020-04-15"),
        domains=["Government", "Web", "Written"],
        task_subtypes=["Conversational retrieval"],
        license="https://huggingface.co/datasets/McGill-NLP/statcan-dialogue-dataset-retrieval/blob/main/LICENSE.md",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
@inproceedings{lu-etal-2023-statcan,
    title = "The {S}tat{C}an Dialogue Dataset: Retrieving Data Tables through Conversations with Genuine Intents",
    author = "Lu, Xing Han  and
      Reddy, Siva  and
      de Vries, Harm",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2304.01412",
    pages = "2799--2829",
}
""",
        descriptive_stats={
            "n_samples": {"dev": 1000, "test": 1011, "corpus": 5907},
            "avg_character_length": {
                "dev": {
                    "english": {
                        "average_document_length": 6535.865413915693,
                        "average_query_length": 6.869244935543278,
                        "num_documents": 5907,
                        "num_queries": 543,
                        "average_relevant_docs_per_query": 1.4714548802946592,
                    },
                    "french": {
                        "average_document_length": 7078.072794988996,
                        "average_query_length": 6.860655737704918,
                        "num_documents": 5907,
                        "num_queries": 122,
                        "average_relevant_docs_per_query": 1.6475409836065573,
                    },
                },
                "test": {
                    "english": {
                        "average_document_length": 6535.865413915693,
                        "average_query_length": 7.650994575045208,
                        "num_documents": 5907,
                        "num_queries": 553,
                        "average_relevant_docs_per_query": 1.573236889692586,
                    },
                    "french": {
                        "average_document_length": 7078.072794988996,
                        "average_query_length": 5.907407407407407,
                        "num_documents": 5907,
                        "num_queries": 108,
                        "average_relevant_docs_per_query": 1.3055555555555556,
                    },
                },
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_statcan_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=list(_LANGS.keys()),
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True

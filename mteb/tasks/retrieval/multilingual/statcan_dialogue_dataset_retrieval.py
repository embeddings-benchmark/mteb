import json

import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLITS = ["dev", "test"]

_LANGS = {
    # <iso_639_3>-<ISO_15924>
    "english": ["eng-Latn"],
    "french": ["fra-Latn"],
}


def _load_statcan_data(
    path: str, langs: list, splits: str, revision: str | None = None
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
                revision=revision,
            )
            corpus_table = datasets.load_dataset(
                path,
                "corpus",
                split=lang,
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


class StatcanDialogueDatasetRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="StatcanDialogueDatasetRetrieval",
        description="A Dataset for Retrieving Data Tables through Conversations with Genuine Intents, available in English and French.",
        dataset={
            "path": "McGill-NLP/statcan-dialogue-dataset-retrieval",
            "revision": "7a26938c93e99e0759a1df416896bb72527e2f33",
        },
        type="Retrieval",
        category="t2t",
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
        bibtex_citation=r"""
@inproceedings{lu-etal-2023-statcan,
  address = {Dubrovnik, Croatia},
  author = {Lu, Xing Han  and
Reddy, Siva  and
de Vries, Harm},
  booktitle = {Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
  month = may,
  pages = {2799--2829},
  publisher = {Association for Computational Linguistics},
  title = {The {S}tat{C}an Dialogue Dataset: Retrieving Data Tables through Conversations with Genuine Intents},
  url = {https://arxiv.org/abs/2304.01412},
  year = {2023},
}
""",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_statcan_data(
            path=self.metadata.dataset["path"],
            langs=list(_LANGS.keys()),
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True

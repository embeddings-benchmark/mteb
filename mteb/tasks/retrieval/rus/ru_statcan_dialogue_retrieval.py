import json

import datasets

from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLITS = ["dev", "test"]

_LANGS = {
    # <iso_639_3>-<ISO_15924>
    "russian": ["rus-Cyrl"]
}


def _load_statcan_data(path: str, langs: list, splits: list[str], revision: str):
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
                query = json.loads(row["query_ru"])
                query = [
                    {**d, "content_en": d["content"], "content": d["content_ru"]}
                    for d in query
                ]
                query_id = row["query_id"]
                doc_id = row["doc_id"]
                queries[lang][split][query_id] = query
                if query_id not in relevant_docs[lang][split]:
                    relevant_docs[lang][split][query_id] = {}
                relevant_docs[lang][split][query_id][doc_id] = 1

            for row in corpus_table:
                doc_id = row["doc_id"]
                doc_content = row["doc_ru"]
                corpus[lang][split][doc_id] = {"text": doc_content}

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class RuStatcanDialogueDatasetRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RuStatcanDialogueDatasetRetrieval",
        description="A Dataset for Retrieving Data Tables through Conversations with Genuine Intents, available in English and French.",
        dataset={
            "path": "DeepPavlov/statcan-dialogue-dataset-retrieval-ru",
            "revision": "e04590ad2c536acc732b4a35015e161757126339",
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
        sample_creation="machine-translated and verified",
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
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_statcan_data(
            path=self.metadata.dataset["path"],
            langs=list(_LANGS.keys()),
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True

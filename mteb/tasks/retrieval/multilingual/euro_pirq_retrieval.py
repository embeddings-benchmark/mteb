import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_HF_LANG_MAP = {
    "en": "english",
    "fi": "finnish",
    "pt": "portuguese",
}

_LANGUAGES = {
    "en": ["eng-Latn"],
    "fi": ["fin-Latn"],
    "pt": ["por-Latn"],
}


def _load_data_retrieval(
    path: str, langs: list, splits: list, revision: str | None = None
):
    corpus = {lang: {split: {} for split in splits} for lang in langs}
    queries = {lang: {split: {} for split in splits} for lang in langs}
    relevant_docs = {lang: {split: {} for split in splits} for lang in langs}

    for lang in langs:
        hf_lang = _HF_LANG_MAP[lang]

        query_data = datasets.load_dataset(
            path, f"{hf_lang}_queries", revision=revision
        )["queries"]

        for row in query_data:
            q_id = str(row["id"])
            doc_id = str(row["chunk_id"])

            queries[lang]["test"][q_id] = row["query"]

            if q_id not in relevant_docs[lang]["test"]:
                relevant_docs[lang]["test"][q_id] = {}
            relevant_docs[lang]["test"][q_id][doc_id] = 1

        corpus_data = datasets.load_dataset(
            path, f"{hf_lang}_corpus", revision=revision
        )["corpus"]

        for row in corpus_data:
            d_id = str(row["id"])
            corpus[lang]["test"][d_id] = {"title": "", "text": row["content"]}

    return corpus, queries, relevant_docs


class EuroPIRQRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EuroPIRQRetrieval",
        description="The EuroPIRQ retrieval dataset is a multilingual collection designed for evaluating retrieval and cross-lingual retrieval tasks. Dataset contains 10,000 parallel passages & 100 parallel queries (synthetic) in three languages: English, Portuguese, and Finnish, constructed from the European Union's DGT-Acquis corpus.",
        reference="https://huggingface.co/datasets/eherra/EuroPIRQ-retrieval",
        dataset={
            "path": "eherra/EuroPIRQ-retrieval",
            "revision": "28acf331e325acc6f2002c658a8a748c2a499c23",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2025-12-01", "2025-12-31"),
        domains=["Legal"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="found",
        is_public=True,
        bibtex_citation=r"""
@misc{eherra_2025_europirq,
  author = { {Elias H.} },
  publisher = { Hugging Face },
  title = { EuroPIRQ: European Parallel Information Retrieval Queries },
  url = { https://huggingface.co/datasets/eherra/EuroPIRQ-retrieval },
  year = {2025},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data_retrieval(
            path=self.metadata.dataset["path"],
            langs=self.hf_subsets,
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True

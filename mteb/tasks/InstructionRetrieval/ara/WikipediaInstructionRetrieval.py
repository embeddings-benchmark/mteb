from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval

_EVAL_SPLIT = "dev"

_LANGUAGES = {
    "ar": ["ara-Arab"],
}


def _load_wikipedia_data(
    path: str, langs: list, splits: str, cache_dir: str = None, revision: str = None
):
    corpus = {lang: {split: None for split in splits} for lang in langs}
    queries = {lang: {split: None for split in splits} for lang in langs}
    relevant_docs = {lang: {split: None for split in splits} for lang in langs}

    split = _EVAL_SPLIT

    for lang in langs:
        # Load corpus data (Can be several millions for languages)
        corpus_identifier = f"corpus-{lang}"
        corpus_data = datasets.load_dataset(
            path,
            corpus_identifier,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )
        corpus[lang][split] = {}
        for row in corpus_data["corpus"]:
            docid = row["docid"]
            doc_title = row["title"]
            doc_text = row["text"]
            corpus[lang][split][docid] = {"title": doc_title, "text": doc_text}

        # Load queries data
        queries_identifier = f"queries-{lang}"
        queries_data = datasets.load_dataset(
            path,
            queries_identifier,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )
        queries[lang][split] = {}
        for row in queries_data["queries"]:
            query_id = row["query_id"]
            query_text = row["query"]
            queries[lang][split][query_id] = query_text

        # Load relevant documents data
        qrels_identifier = f"{lang}"
        qrels_data = datasets.load_dataset(
            path,
            qrels_identifier,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )
        relevant_docs[lang][split] = {}
        for row in qrels_data[split]:
            query_id = row["query_id"]
            doc_id = row["docid"]
            score = row["score"]
            if query_id not in relevant_docs[lang][split]:
                relevant_docs[lang][split][query_id] = {}
            relevant_docs[lang][split][query_id][doc_id] = score

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs



class WikipediaInstructionRetrieval(AbsTaskInstructionRetrieval):
    metadata = TaskMetadata(
        name="WikipediaInstructionRetrieval",
        description="This dataset contains a pre-processed version from Wikipedia suitable for semantic search.",
        reference="https://huggingface.co/datasets/Cohere/wikipedia-22-12-ar-embeddings",
        dataset={
            "path": "Cohere/wikipedia-22-12-ar-embeddings",
            "revision": "ea5f00014bd7626aa55affb07de57d519ab3309a",
        },
        type="InstructionRetrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGUAGES,
        main_score="p-MRR",
        date=("2023-01-14", "2024-03-22"),
        form=["written"],
        domains=["Web"],
        task_subtypes=[],
        license="Not specified",
        socioeconomic_status="low",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation=""" """,
        n_samples=None,
        avg_character_length=None,
    )
    
    def __init__(self):
        super().__init__()
        self.langs = list(_LANGUAGES.keys())

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_wikipedia_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.langs,
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
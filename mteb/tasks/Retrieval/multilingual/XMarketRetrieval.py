from __future__ import annotations

import datasets

from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

_EVAL_SPLIT = "test"

_EVAL_LANGS = {
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
}


def _load_xmarket_data(
    path: str, langs: list, split: str, cache_dir: str = None, revision: str = None
):
    corpus = {lang: {split: None} for lang in langs}
    queries = {lang: {split: None} for lang in langs}
    relevant_docs = {lang: {split: None} for lang in langs}

    for lang in langs:
        corpus_rows = datasets.load_dataset(
            path,
            f"corpus-{lang}",
            languages=[lang],
            split=split,
            cache_dir=cache_dir,
        )
        query_rows = datasets.load_dataset(
            path,
            f"queries-{lang}",
            languages=[lang],
            revision=revision,
            split=split,
            cache_dir=cache_dir,
        )
        qrels_rows = datasets.load_dataset(
            path,
            f"qrels-{lang}",
            languages=[lang],
            revision=revision,
            split=split,
            cache_dir=cache_dir,
        )

        corpus[lang][split] = {row["_id"]: row for row in corpus_rows}
        queries[lang][split] = {row["_id"]: row["text"] for row in query_rows}
        relevant_docs[lang][split] = {
            row["_id"]: {v: 1 for v in row["text"].split(" ")} for row in qrels_rows
        }

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class XMarket(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="XMarket",
        description="XMarket",
        reference=None,
        dataset={
            "path": "jinaai/xmarket_ml",
            "revision": "dfe57acff5b62c23732a7b7d3e3fb84ff501708b",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@inproceedings{Bonab_2021, series={CIKM ’21},
   title={Cross-Market Product Recommendation},
   url={http://dx.doi.org/10.1145/3459637.3482493},
   DOI={10.1145/3459637.3482493},
   booktitle={Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management},
   publisher={ACM},
   author={Bonab, Hamed and Aliannejadi, Mohammad and Vardasbi, Ali and Kanoulas, Evangelos and Allan, James},
   year={2021},
   month=oct, collection={CIKM ’21} }""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "de": {
                        "average_document_length": 187.4061197288943,
                        "average_query_length": 15.717612088184294,
                        "num_documents": 70526,
                        "num_queries": 4037,
                        "average_relevant_docs_per_query": 54.3522417636859,
                    },
                    "en": {
                        "average_document_length": 452.792089662076,
                        "average_query_length": 15.881635344543357,
                        "num_documents": 218777,
                        "num_queries": 9099,
                        "average_relevant_docs_per_query": 85.43719090009891,
                    },
                    "es": {
                        "average_document_length": 279.67909262759923,
                        "average_query_length": 19.97062937062937,
                        "num_documents": 39675,
                        "num_queries": 3575,
                        "average_relevant_docs_per_query": 36.01006993006993,
                    },
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_xmarket_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.metadata.eval_langs,
            split=self.metadata_dict["eval_splits"][0],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True

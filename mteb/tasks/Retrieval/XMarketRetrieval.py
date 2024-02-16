from ...abstasks import MultilingualTask
from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval

import datasets

_EVAL_SPLIT = "test"
_EVAL_LANGS = ["es", "de", "en"]


def _load_xmarket_data(path: str, langs: list, split: str, cache_dir: str=None, revision: str=None):
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
        relevant_docs[lang][split] = {row["_id"]: {v: 1 for v in row["text"].split(" ")} for row in qrels_rows}

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class XMarketDE(MultilingualTask, AbsTaskRetrieval):

    @property
    def description(self):
        return {
            "name": "XMarket",
            "hf_hub_name": "jinaai/xmarket_ml",
            "description": "XMarket is an ecommerce category to product retrieval dataset in German.",
            "reference": "https://xmrec.github.io/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": [_EVAL_SPLIT],
            "eval_langs": _EVAL_LANGS,
            "main_score": "ndcg_at_10",
            "revision": "dfe57acff5b62c23732a7b7d3e3fb84ff501708b",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_xmarket_data(
            path=self.description['hf_hub_name'],
            langs=self.langs,
            split=self.description['eval_splits'][0],
            cache_dir=kwargs.get('cache_dir', None),
            revision=self.description['revision'],
        )

        self.data_loaded = True

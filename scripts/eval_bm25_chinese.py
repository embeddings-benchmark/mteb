"""Compare BM25 variants on LeCaRDv2 (Chinese legal retrieval).

Models:
  1. bm25s                   – English stemmer/stopwords (baseline)
  2. bm25s-multilingual      – XLM-R subword tokenizer, no stopwords
  3. bm25s-multilingual-exp1 – XLM-R + high-frequency token stopwords (≥90% of docs)
  4. bm25s-unicode-exp2      – script-aware Unicode tokenizer, no ML model
  5. bm25s-lang-aware-exp3   – language-aware stemmer/stopwords derived from task metadata
"""

from __future__ import annotations

import logging

import mteb
from mteb import ResultCache
from mteb.models.model_implementations.bm25 import (
    BM25MultilingualSearch,
    BM25UnicodeSplitSearch,
    BM25Search,
)

logger = logging.getLogger(__name__)
CACHE = ResultCache(cache_path="results/bm25-chinese-eval")
TASKS = mteb.get_tasks(tasks=["LeCaRDv2"])


class BM25MultilingualWithFreqStopwords(BM25MultilingualSearch):
    """BM25 multilingual with corpus-frequency-based stopwords.

    Tokens appearing in >= `freq_threshold` fraction of corpus documents are
    treated as stopwords and removed from both corpus and query token lists.
    """

    def __init__(self, freq_threshold: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.freq_threshold = freq_threshold
        self._stopwords: set[str] = set()

    def _tokenize_raw(self, texts: list[str]) -> list[list[str]]:
        token_lists = []
        for text in texts:
            raw = self.hf_tokenizer.encode(text, add_special_tokens=False).tokens
            clean = [
                t.replace(" ", "").replace("▁", "")
                for t in raw
                if t.replace(" ", "").replace("▁", "")
            ]
            token_lists.append(clean)
        return token_lists

    def index(self, corpus, *, task_metadata, hf_split, hf_subset, encode_kwargs, num_proc=None):
        corpus_texts = [
            "\n".join([doc.get("title", ""), doc["text"]]) for doc in corpus
        ]
        n_docs = len(corpus_texts)
        token_lists = self._tokenize_raw(corpus_texts)

        doc_freq: dict[str, int] = {}
        for tokens in token_lists:
            for t in set(tokens):
                doc_freq[t] = doc_freq.get(t, 0) + 1

        self._stopwords = {t for t, df in doc_freq.items() if df / n_docs >= self.freq_threshold}
        logger.info(
            f"Freq-stopwords: {len(self._stopwords)} tokens removed (threshold={self.freq_threshold})"
        )

        super().index(
            corpus,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            encode_kwargs=encode_kwargs,
            num_proc=num_proc,
        )

    def _encode(self, texts: list[str]):
        from bm25s.tokenization import Tokenized

        token_lists = self._tokenize_raw(texts)

        vocab: dict[str, int] = {}
        encoded_ids = []
        for tokens in token_lists:
            ids = []
            for t in tokens:
                if t in self._stopwords:
                    continue
                if t not in vocab:
                    vocab[t] = len(vocab)
                ids.append(vocab[t])
            encoded_ids.append(ids)

        return Tokenized(ids=encoded_ids, vocab=vocab)


def make_exp1_model():
    """Clone the multilingual ModelMeta, swap the loader, rename to *-exp1."""
    base = mteb.get_model_meta("mteb/baseline-bm25s-multilingual")
    return base.model_copy(
        update={
            "name": "mteb/baseline-bm25s-multilingual-exp1",
            "loader": lambda model_name, **kwargs: BM25MultilingualWithFreqStopwords(**kwargs),
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    exp2_meta = mteb.get_model_meta("mteb/baseline-bm25s-multilingual").model_copy(
        update={
            "name": "mteb/baseline-bm25s-unicode-exp2",
            "loader": lambda model_name, **kwargs: BM25UnicodeSplitSearch(**kwargs),
        }
    )

    exp3_meta = mteb.get_model_meta("mteb/baseline-bm25s-lang-aware").model_copy(
        update={"name": "mteb/baseline-bm25s-lang-aware-jieba-exp3"}
    )

    models = [
        mteb.get_model_meta("mteb/baseline-bm25s"),
        mteb.get_model_meta("mteb/baseline-bm25s-multilingual"),
        make_exp1_model(),
        exp2_meta,
        exp3_meta,
    ]

    for meta in models:
        print(f"\n{'='*60}")
        print(f"Running: {meta.name}")
        print(f"{'='*60}")
        model = meta.load_model()
        results = mteb.evaluate(model, TASKS, cache=CACHE)
        for r in results.task_results:
            score = r.get_score()
            print(f"  {r.task_name}: ndcg@10 = {score:.4f}")

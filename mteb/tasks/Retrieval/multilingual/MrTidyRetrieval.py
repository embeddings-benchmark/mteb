from __future__ import annotations

import logging

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_LANGS = {
    "bengali": ["ben-Beng"],
    "english": ["eng-Latn"],
    "finnish": ["fin-Latn"],
    "russian": ["rus-Cyrl"],
    "korean": ["kor-Kore"],
    "japanese": ["jpn-Jpan"],
    "telugu": ["tel-Telu"],
    "thai": ["tha-Thai"],
    "swahili": ["swa-Latn"],
    "arabic": ["ara-Arab"],
    "indonesian": ["ind-Latn"],
}
_EVAL_SPLIT = "test"

logger = logging.getLogger(__name__)


def _load_data_retrieval(
    path: str, langs: list, splits: str, cache_dir: str = None, revision: str = None
):
    corpus = {lang: {split: {} for split in splits} for lang in langs}
    queries = {lang: {split: {} for split in splits} for lang in langs}
    relevant_docs = {lang: {split: {} for split in splits} for lang in langs}

    split = _EVAL_SPLIT

    for lang in langs:
        qrels_data = datasets.load_dataset(
            path,
            name=f"{lang}-qrels",
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )[split]

        for row in qrels_data:
            query_id = row["query-id"]
            doc_id = row["corpus-id"]
            score = row["score"]
            if query_id not in relevant_docs[lang][split]:
                relevant_docs[lang][split][query_id] = {}
            relevant_docs[lang][split][query_id][doc_id] = score

        corpus_data = datasets.load_dataset(
            path,
            name=f"{lang}-corpus",
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )["train"]

        for row in corpus_data:
            doc_id = row["_id"]
            doc_title = row["title"]
            doc_text = row["text"]
            corpus[lang][split][doc_id] = {"title": doc_title, "text": doc_text}

        queries_data = datasets.load_dataset(
            path,
            name=f"{lang}-queries",
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
        )[split]

        for row in queries_data:
            query_id = row["_id"]
            query_text = row["text"]
            queries[lang][split][query_id] = query_text

        queries = queries
        logger.info("Loaded %d %s Queries.", len(queries), split.upper())

    return corpus, queries, relevant_docs


class MrTidyRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MrTidyRetrieval",
        description="Mr. TyDi is a multi-lingual benchmark dataset built on TyDi, covering eleven typologically diverse languages. It is designed for monolingual retrieval, specifically to evaluate ranking with learned dense representations.",
        reference="https://huggingface.co/datasets/castorini/mr-tydi",
        dataset={
            "path": "mteb/mrtidy",
            "revision": "fc24a3ce8f09746410daee3d5cd823ff7a0675b7",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2023-11-01", "2024-05-15"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="cc-by-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{mrtydi,
  author = {Xinyu Zhang and Xueguang Ma and Peng Shi and Jimmy Lin},
  journal = {arXiv:2108.08787},
  title = {{Mr. TyDi}: A Multi-lingual Benchmark for Dense Retrieval},
  year = {2021},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data_retrieval(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.hf_subsets,
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True

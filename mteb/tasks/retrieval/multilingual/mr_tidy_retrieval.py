from __future__ import annotations

import logging
from typing import Any

import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

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
    path: str, langs: list, splits: list[str], revision: str | None = None
) -> dict[str, dict[str, RetrievalSplitData]]:
    dataset = {lang: {} for lang in langs}
    for lang in langs:
        corpus = (
            datasets.load_dataset(
                path, name=f"{lang}-corpus", revision=revision, split="train"
            )
            .rename_column("_id", "id")
            .select_columns(["id", "title", "text"])
        )
        for split in splits:
            queries = (
                datasets.load_dataset(
                    path, name=f"{lang}-queries", revision=revision, split=split
                )
                .rename_column("_id", "id")
                .select_columns(["id", "text"])
            )
            qrels = datasets.load_dataset(
                path, name=f"{lang}-qrels", revision=revision, split=split
            )
            relevant_docs = {}
            for row in qrels:
                relevant_docs.setdefault(row["query-id"], {})[row["corpus-id"]] = row[
                    "score"
                ]
            dataset[lang][split] = RetrievalSplitData(
                corpus=corpus,
                queries=queries,
                relevant_docs=relevant_docs,
                top_ranked=None,
            )
    return dataset


class MrTidyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MrTidyRetrieval",
        description="Mr. TyDi is a multi-lingual benchmark dataset built on TyDi, covering eleven typologically diverse languages. It is designed for monolingual retrieval, specifically to evaluate ranking with learned dense representations.",
        reference="https://huggingface.co/datasets/castorini/mr-tydi",
        dataset={
            "path": "mteb/mrtidy",
            "revision": "fc24a3ce8f09746410daee3d5cd823ff7a0675b7",
        },
        type="Retrieval",
        category="t2t",
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
@inproceedings{mrtydi,
  address = {Punta Cana, Dominican Republic},
  author = {Zhang, Xinyu and Ma, Xueguang and Shi, Peng and Lin, Jimmy},
  booktitle = {Proceedings of the 1st Workshop on Multilingual Representation Learning},
  doi = {10.18653/v1/2021.mrl-1.12},
  editor = {Ataman, Duygu and Birch, Alexandra and Conneau, Alexis and Firat, Orhan and Ruder, Sebastian and Sahin, Gozde Gul},
  month = {nov},
  pages = {127--137},
  publisher = {Association for Computational Linguistics},
  title = {Mr. {T}y{D}i: A Multi-lingual Benchmark for Dense Retrieval},
  url = {https://aclanthology.org/2021.mrl-1.12/},
  year = {2021},
}
""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = _load_data_retrieval(
            path=self.metadata.dataset["path"],
            langs=self.hf_subsets,
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True

from __future__ import annotations

from collections import defaultdict

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskRetrieval, MultilingualTask
from ....abstasks.AbsTaskRetrieval import *

_LANGUAGES = {
    "fas": ["fas-Arab"],
    "rus": ["rus-Cyrl"],
    "zho": ["zho-Hans"],
}


def load_neuclir_data(
    path: str,
    langs: list,
    eval_splits: list,
    cache_dir: str = None,
    revision: str = None,
):
    corpus = {lang: {split: None for split in eval_splits} for lang in langs}
    queries = {lang: {split: None for split in eval_splits} for lang in langs}
    relevant_docs = {lang: {split: None for split in eval_splits} for lang in langs}

    for lang in langs:
        lang_corpus = datasets.load_dataset(
            path, f"corpus-{lang}", cache_dir=cache_dir, revision=revision
        )["corpus"]
        lang_queries = datasets.load_dataset(
            path, f"queries-{lang}", cache_dir=cache_dir, revision=revision
        )["queries"]
        lang_qrels = datasets.load_dataset(
            path, f"{lang}", cache_dir=cache_dir, revision=revision
        )["test"]
        corpus[lang] = {"test": {e["_id"]: {"text": e["text"]} for e in lang_corpus}}
        queries[lang] = {"test": {e["_id"]: e["text"] for e in lang_queries}}
        relevant_docs[lang]["test"] = defaultdict(dict)
        for item in lang_qrels:
            relevant_docs[lang]["test"][item["query-id"]].update(
                {item["corpus-id"]: item["score"]}
            )

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)
    return corpus, queries, relevant_docs


class NeuCLIR2022Retrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NeuCLIR2022Retrieval",
        description="The task involves identifying and retrieving the documents that are relevant to the queries.",
        reference="https://neuclir.github.io/",
        dataset={
            "path": "mteb/neuclir-2022",
            "revision": "16a506c748bf2b7fdb886677dcdbbc07c677a7fa",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_20",
        date=("2021-08-01", "2022-06-30"),
        form=["written"],
        domains=["News"],
        task_subtypes=None,
        license="odc-by",
        socioeconomic_status="medium",
        annotations_creators="expert-annotated",
        dialect=None,
        text_creation="found",
        bibtex_citation="""@article{lawrie2023overview,
  title={Overview of the TREC 2022 NeuCLIR track},
  author={Lawrie, Dawn and MacAvaney, Sean and Mayfield, James and McNamee, Paul and Oard, Douglas W and Soldaini, Luca and Yang, Eugene},
  journal={arXiv preprint arXiv:2304.12367},
  year={2023}
}""",
        n_samples=None,
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_neuclir_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.metadata.eval_langs,
            eval_splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )
        self.data_loaded = True

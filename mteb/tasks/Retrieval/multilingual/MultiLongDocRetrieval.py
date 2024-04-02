from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskRetrieval, MultilingualTask
from ....abstasks.AbsTaskRetrieval import *

_LANGUAGES = [
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "it",
    "ja",
    "ko",
    "pt",
    "ru",
    "th",
    "zh",
]


def load_mldr_data(
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
        lang_corpus = {e["docid"]: {"text": e["text"]} for e in lang_corpus}
        lang_data = datasets.load_dataset(path, lang, cache_dir=cache_dir)
        for split in eval_splits:
            corpus[lang][split] = lang_corpus
            queries[lang][split] = {e["query_id"]: e["query"] for e in lang_data[split]}
            relevant_docs[lang][split] = {
                e["query_id"]: {e["positive_passages"][0]["docid"]: 1}
                for e in lang_data[split]
            }

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)
    return corpus, queries, relevant_docs


class MultiLongDocRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultiLongDocRetrieval",
        description="MultiLongDocRetrieval",
        reference="https://arxiv.org/abs/2402.03216",
        dataset={
            "path": "Shitao/MLDR",
            "revision": "d79af07e969a6678fcbbe819956840425816468f",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["dev", "test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@misc{bge-m3,
      title={BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
      author={Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu},
      year={2024},
      eprint={2402.03216},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
""",
        n_samples=None,
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_mldr_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.metadata.eval_langs,
            eval_splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )
        self.data_loaded = True

from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import *

_LANGUAGES = {
    "ar": ["ara-Arab"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Jpan"],
    "ko": ["kor-Hang"],
    "pt": ["por-Latn"],
    "ru": ["rus-Cyrl"],
    "th": ["tha-Thai"],
    "zh": ["cmn-Hans"],
}


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
            path,
            f"corpus-{lang}",
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=True,
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
        description="""Multi Long Doc Retrieval (MLDR) 'is curated by the multilingual articles from Wikipedia, Wudao and mC4 (see Table 7), and NarrativeQA (Kocˇisky ́ et al., 2018; Gu ̈nther et al., 2023), which is only for English.' (Chen et al., 2024).
        It is constructed by sampling lengthy articles from Wikipedia, Wudao and mC4 datasets and randomly choose paragraphs from them. Then we use GPT-3.5 to generate questions based on these paragraphs. The generated question and the sampled article constitute a new text pair to the dataset.""",
        reference="https://arxiv.org/abs/2402.03216",  # also: https://huggingface.co/datasets/Shitao/MLDR
        dataset={
            "path": "Shitao/MLDR",
            "revision": "d67138e705d963e346253a80e59676ddb418810a",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev", "test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=(
            "2000-01-01",
            "2024-12-31",
        ),  # Not found in the paper, guessed using the paper's publication date and constituent datasets
        domains=[
            "Encyclopaedic",
            "Written",
            "Web",
            "Non-fiction",
            "Fiction",
        ],  # narrativeqa, wikipedia, wudao, mC4
        task_subtypes=[],
        license="mit",
        annotations_creators="LM-generated",  # gpt-3.5
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{bge-m3,
      title={BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
      author={Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu},
      year={2024},
      eprint={2402.03216},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
""",
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

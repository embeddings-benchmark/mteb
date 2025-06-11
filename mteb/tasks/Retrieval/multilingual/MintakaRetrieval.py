from __future__ import annotations

import datasets

from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

_EVAL_SPLIT = "test"
_LANGS = {
    "ar": ["ara-Arab"],
    "de": ["deu-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Hira"],
    "pt": ["por-Latn"],
}


def _load_mintaka_data(
    path: str,
    langs: list,
    split: str,
    trust_remote_code: bool,
    cache_dir: str = None,
    revision: str = None,
):
    queries = {lang: {split: {}} for lang in langs}
    corpus = {lang: {split: {}} for lang in langs}
    relevant_docs = {lang: {split: {}} for lang in langs}

    for lang in langs:
        data = datasets.load_dataset(
            path,
            lang,
            split=split,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        question_ids = {
            question: _id for _id, question in enumerate(set(data["question"]))
        }
        answer_ids = {answer: _id for _id, answer in enumerate(set(data["answer"]))}

        for row in data:
            question = row["question"]
            answer = row["answer"]
            query_id = f"Q{question_ids[question]}"
            queries[lang][split][query_id] = question
            doc_id = f"D{answer_ids[answer]}"
            corpus[lang][split][doc_id] = {"text": answer}
            if query_id not in relevant_docs[lang][split]:
                relevant_docs[lang][split][query_id] = {}
            relevant_docs[lang][split][query_id][doc_id] = 1

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class MintakaRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MintakaRetrieval",
        description="We introduce Mintaka, a complex, natural, and multilingual dataset designed for experimenting with end-to-end question-answering models. Mintaka is composed of 20,000 question-answer pairs collected in English, annotated with Wikidata entities, and translated into Arabic, French, German, Hindi, Italian, Japanese, Portuguese, and Spanish for a total of 180,000 samples. Mintaka includes 8 types of complex questions, including superlative, intersection, and multi-hop questions, which were naturally elicited from crowd workers. ",
        reference=None,
        dataset={
            "path": "jinaai/mintakaqa",
            "revision": "efa78cc2f74bbcd21eff2261f9e13aebe40b814e",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-01-01"),  # best guess: based on the date of the paper
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="derived",  # best guess
        dialect=[],
        sample_creation="human-translated",
        bibtex_citation=r"""
@inproceedings{sen-etal-2022-mintaka,
  address = {Gyeongju, Republic of Korea},
  author = {Sen, Priyanka  and
Aji, Alham Fikri  and
Saffari, Amir},
  booktitle = {Proceedings of the 29th International Conference on Computational Linguistics},
  month = oct,
  pages = {1604--1619},
  publisher = {International Committee on Computational Linguistics},
  title = {Mintaka: A Complex, Natural, and Multilingual Dataset for End-to-End Question Answering},
  url = {https://aclanthology.org/2022.coling-1.138},
  year = {2022},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_mintaka_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.metadata.eval_langs,
            split=self.metadata_dict["eval_splits"][0],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
            trust_remote_code=self.metadata_dict["dataset"]["trust_remote_code"],
        )

        self.data_loaded = True

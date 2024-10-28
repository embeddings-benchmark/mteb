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
        bibtex_citation="""@inproceedings{sen-etal-2022-mintaka,
    title = "Mintaka: A Complex, Natural, and Multilingual Dataset for End-to-End Question Answering",
    author = "Sen, Priyanka  and
      Aji, Alham Fikri  and
      Saffari, Amir",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.138",
    pages = "1604--1619"
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "ar": {
                        "average_document_length": 12.736418511066399,
                        "average_query_length": 55.275533363595095,
                        "num_documents": 1491,
                        "num_queries": 2203,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "de": {
                        "average_document_length": 14.40060422960725,
                        "average_query_length": 65.41322662173546,
                        "num_documents": 1655,
                        "num_queries": 2374,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "es": {
                        "average_document_length": 14.291789722386296,
                        "average_query_length": 64.88325082508251,
                        "num_documents": 1693,
                        "num_queries": 2424,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "fr": {
                        "average_document_length": 14.407234539089849,
                        "average_query_length": 68.88452088452088,
                        "num_documents": 1714,
                        "num_queries": 2442,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "hi": {
                        "average_document_length": 12.71038961038961,
                        "average_query_length": 58.404637247569184,
                        "num_documents": 770,
                        "num_queries": 1337,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "it": {
                        "average_document_length": 14.365985576923077,
                        "average_query_length": 64.39707724425887,
                        "num_documents": 1664,
                        "num_queries": 2395,
                        "average_relevant_docs_per_query": 1.0004175365344468,
                    },
                    "ja": {
                        "average_document_length": 9.167713567839195,
                        "average_query_length": 29.961937716262977,
                        "num_documents": 1592,
                        "num_queries": 2312,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "pt": {
                        "average_document_length": 14.244471744471744,
                        "average_query_length": 60.42225998300765,
                        "num_documents": 1628,
                        "num_queries": 2354,
                        "average_relevant_docs_per_query": 1.0004248088360237,
                    },
                }
            },
        },
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

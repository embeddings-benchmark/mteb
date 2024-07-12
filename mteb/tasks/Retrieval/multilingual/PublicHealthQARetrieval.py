from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

_EVAL_SPLIT = "test"

_LANGS = {
    # <iso_639_3>-<ISO_15924>
    "arabic": ["ara-Arab"],
    "chinese": ["zho-Hans"],
    "english": ["eng-Latn"],
    "french": ["fra-Latn"],
    "korean": ["kor-Hang"],
    "russian": ["rus-Cyrl"],
    "spanish": ["spa-Latn"],
    "vietnamese": ["vie-Latn"],
}


def _load_publichealthqa_data(
    path: str, langs: list, split: str, cache_dir: str = None, revision: str = None
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


class PublicHealthQARetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PublicHealthQA",
        description="A multilingual dataset for public health question answering, based on FAQ sourced from CDC and WHO.",
        dataset={
            "path": "xhluca/publichealth-qa",
            "revision": "main",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        reference="https://huggingface.co/datasets/xhluca/publichealth-qa",
        date=("2020-01-01", "2020-04-15"),
        domains=["Medical", "Government", "Web", "Written"],
        task_subtypes=["Question answering"],
        license="CC BY-NC-SA 3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
@misc {xing_han_lu_2024,
	author       = { {Xing Han Lu} },
	title        = { publichealth-qa (Revision 3b67b6b) },
	year         = 2024,
	url          = { https://huggingface.co/datasets/xhluca/publichealth-qa },
	doi          = { 10.57967/hf/2247 },
	publisher    = { Hugging Face }
}
""",
        descriptive_stats={
            "n_samples": {"test": 888},
            "avg_character_length": {
                "test": {
                    "arabic": {
                        "average_document_length": 836.8850574712644,
                        "average_query_length": 79.84883720930233,
                        "num_documents": 87,
                        "num_queries": 87,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "chinese": {
                        "average_document_length": 239.58282208588957,
                        "average_query_length": 24.828220858895705,
                        "num_documents": 163,
                        "num_queries": 163,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "english": {
                        "average_document_length": 799.3430232558139,
                        "average_query_length": 71.78488372093024,
                        "num_documents": 172,
                        "num_queries": 172,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "french": {
                        "average_document_length": 1021.6823529411764,
                        "average_query_length": 101.88235294117646,
                        "num_documents": 85,
                        "num_queries": 85,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "korean": {
                        "average_document_length": 339.0,
                        "average_query_length": 36.90909090909091,
                        "num_documents": 77,
                        "num_queries": 77,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "russian": {
                        "average_document_length": 985.1076923076923,
                        "average_query_length": 85.2,
                        "num_documents": 65,
                        "num_queries": 65,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "spanish": {
                        "average_document_length": 941.1666666666666,
                        "average_query_length": 84.67901234567901,
                        "num_documents": 162,
                        "num_queries": 162,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "vietnamese": {
                        "average_document_length": 704.5454545454545,
                        "average_query_length": 71.83116883116882,
                        "num_documents": 77,
                        "num_queries": 77,
                        "average_relevant_docs_per_query": 1.0,
                    },
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_publichealthqa_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.hf_subsets,
            split=self.metadata_dict["eval_splits"][0],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True

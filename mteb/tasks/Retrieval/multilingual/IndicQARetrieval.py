from __future__ import annotations

from hashlib import sha256

import datasets

from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

_LANGUAGES = {
    "as": ["asm-Beng"],
    "bn": ["ben-Beng"],
    "gu": ["guj-Gujr"],
    "hi": ["hin-Deva"],
    "kn": ["kan-Knda"],
    "ml": ["mal-Mlym"],
    "mr": ["mar-Deva"],
    "or": ["ory-Orya"],
    "pa": ["pan-Guru"],
    "ta": ["tam-Taml"],
    "te": ["tel-Telu"],
}


class IndicQARetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="IndicQARetrieval",
        dataset={
            "path": "ai4bharat/IndicQA",
            "revision": "570d90ae4f7b64fe4fdd5f42fc9f9279b8c9fd9d",
            "trust_remote_code": True,
        },
        description="IndicQA is a manually curated cloze-style reading comprehension dataset that can be used for evaluating question-answering models in 11 Indic languages. It is repurposed retrieving relevant context for each question.",
        reference="https://arxiv.org/abs/2212.05409",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2022-08-01", "2022-12-20"),
        domains=["Web", "Written"],
        task_subtypes=[],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="""@article{doddapaneni2022towards,
  title     = {Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages},
  author    = {Sumanth Doddapaneni and Rahul Aralikatte and Gowtham Ramesh and Shreyansh Goyal and Mitesh M. Khapra and Anoop Kunchukuttan and Pratyush Kumar},
  journal   = {Annual Meeting of the Association for Computational Linguistics},
  year      = {2022},
  doi       = {10.18653/v1/2023.acl-long.693}
}""",
        descriptive_stats={
            "n_samples": {"test": 18586},
            "avg_character_length": {
                "test": {
                    "as": {
                        "average_document_length": 1401.28,
                        "average_query_length": 56.60504201680672,
                        "num_documents": 250,
                        "num_queries": 1785,
                        "average_relevant_docs_per_query": 1.0016806722689076,
                    },
                    "bn": {
                        "average_document_length": 2196.012,
                        "average_query_length": 57.069239500567534,
                        "num_documents": 250,
                        "num_queries": 1762,
                        "average_relevant_docs_per_query": 1.0005675368898979,
                    },
                    "gu": {
                        "average_document_length": 960.4959677419355,
                        "average_query_length": 60.3712158808933,
                        "num_documents": 248,
                        "num_queries": 2015,
                        "average_relevant_docs_per_query": 1.0009925558312656,
                    },
                    "hi": {
                        "average_document_length": 2550.770114942529,
                        "average_query_length": 52.84909326424871,
                        "num_documents": 261,
                        "num_queries": 1544,
                        "average_relevant_docs_per_query": 1.0019430051813472,
                    },
                    "kn": {
                        "average_document_length": 882.7354085603113,
                        "average_query_length": 50.58734344100198,
                        "num_documents": 257,
                        "num_queries": 1517,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "ml": {
                        "average_document_length": 2522.6437246963565,
                        "average_query_length": 75.93635790800252,
                        "num_documents": 247,
                        "num_queries": 1587,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "mr": {
                        "average_document_length": 1711.74,
                        "average_query_length": 58.785,
                        "num_documents": 250,
                        "num_queries": 1600,
                        "average_relevant_docs_per_query": 1.0,
                    },
                    "or": {
                        "average_document_length": 801.9206349206349,
                        "average_query_length": 55.072792362768496,
                        "num_documents": 252,
                        "num_queries": 1676,
                        "average_relevant_docs_per_query": 1.0011933174224343,
                    },
                    "pa": {
                        "average_document_length": 1423.5062240663901,
                        "average_query_length": 58.394925178919976,
                        "num_documents": 241,
                        "num_queries": 1537,
                        "average_relevant_docs_per_query": 1.0013012361743656,
                    },
                    "ta": {
                        "average_document_length": 2288.2608695652175,
                        "average_query_length": 54.06211869107044,
                        "num_documents": 253,
                        "num_queries": 1803,
                        "average_relevant_docs_per_query": 1.0005546311702718,
                    },
                    "te": {
                        "average_document_length": 2936.176,
                        "average_query_length": 67.00634371395617,
                        "num_documents": 250,
                        "num_queries": 1734,
                        "average_relevant_docs_per_query": 1.0,
                    },
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        split = "test"
        queries = {lang: {split: {}} for lang in self.hf_subsets}
        corpus = {lang: {split: {}} for lang in self.hf_subsets}
        relevant_docs = {lang: {split: {}} for lang in self.hf_subsets}

        for lang in self.hf_subsets:
            data = datasets.load_dataset(
                name=f"indicqa.{lang}", **self.metadata_dict["dataset"]
            )[split]
            data = data.filter(lambda x: x["answers"]["text"] != "")

            question_ids = {
                question: sha256(question.encode("utf-8")).hexdigest()
                for question in set(data["question"])
            }
            context_ids = {
                context: sha256(context.encode("utf-8")).hexdigest()
                for context in set(data["context"])
            }

            for row in data:
                question = row["question"]
                context = row["context"]
                query_id = question_ids[question]
                queries[lang][split][query_id] = question

                doc_id = context_ids[context]
                corpus[lang][split][doc_id] = {"text": context}
                if query_id not in relevant_docs[lang][split]:
                    relevant_docs[lang][split][query_id] = {}
                relevant_docs[lang][split][query_id][doc_id] = 1

        self.corpus = datasets.DatasetDict(corpus)
        self.queries = datasets.DatasetDict(queries)
        self.relevant_docs = datasets.DatasetDict(relevant_docs)

        self.data_loaded = True

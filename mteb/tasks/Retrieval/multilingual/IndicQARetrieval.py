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

from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

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

SPLIT = "test"


class IndicQARetrieval(AbsTaskRetrieval, MultilingualTask):
    metadata = TaskMetadata(
        name="IndicQARetrieval",
        dataset={
            "path": "mteb/IndicQARetrieval",
            "revision": "51e8b328988795d658f6f34acd34044e9346e2ee",
        },
        description="IndicQA is a manually curated cloze-style reading comprehension dataset that can be used for evaluating question-answering models in 11 Indic languages. It is repurposed retrieving relevant context for each question.",
        reference="https://arxiv.org/abs/2212.05409",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[SPLIT],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2022-08-01", "2022-12-20"),
        domains=["Web", "Written"],
        task_subtypes=[],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@article{doddapaneni2022towards,
  author = {Sumanth Doddapaneni and Rahul Aralikatte and Gowtham Ramesh and Shreyansh Goyal and Mitesh M. Khapra and Anoop Kunchukuttan and Pratyush Kumar},
  doi = {10.18653/v1/2023.acl-long.693},
  journal = {Annual Meeting of the Association for Computational Linguistics},
  title = {Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages},
  year = {2022},
}
""",
    )

    def load_data(self, **kwargs):
        self.corpus = {
            subset: {split: {} for split in self.eval_splits}
            for subset in self.hf_subsets
        }
        self.queries = {
            subset: {split: {} for split in self.eval_splits}
            for subset in self.hf_subsets
        }
        self.relevant_docs = {
            subset: {split: {} for split in self.eval_splits}
            for subset in self.hf_subsets
        }
        for lang in self.hf_subsets:
            corpus = load_dataset(
                self.metadata.dataset["path"],
                name=f"{lang}-corpus",
                split=SPLIT,
                revision=self.metadata.dataset["revision"],
            )

            self.corpus[lang][SPLIT] = {
                row["_id"]: {
                    "text": row["text"],
                    "title": row.get("title", ""),
                }
                for row in corpus
            }

            queries = load_dataset(
                self.metadata.dataset["path"],
                name=f"{lang}-queries",
                split=SPLIT,
                revision=self.metadata.dataset["revision"],
            )

            self.queries[lang][SPLIT] = {row["_id"]: row["text"] for row in queries}

            qrels = load_dataset(
                self.metadata.dataset["path"],
                name=f"{lang}-qrels",
                split=SPLIT,
                revision=self.metadata.dataset["revision"],
            )

            for row in qrels:
                if row["query-id"] not in self.relevant_docs[lang][SPLIT]:
                    self.relevant_docs[lang][SPLIT][row["query-id"]] = {}
                self.relevant_docs[lang][SPLIT][row["query-id"]][row["corpus-id"]] = (
                    row["score"]
                )

        self.data_loaded = True

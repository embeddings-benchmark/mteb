from __future__ import annotations

import datasets

from mteb.abstasks import MultilingualTask
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGS = [
    "python",
    "javascript",
    "typescript",
    "go",
    "ruby",
    "java",
    "php",
    "c",
    "c++",
    "rust",
    "swift",
    "scala",
    "shell",
]


class CodeEditSearchRetrieval(MultilingualTask, AbsTaskRetrieval):
    _EVAL_SPLIT = "train"
    metadata = TaskMetadata(
        name="CodeEditSearchRetrieval",
        description="The dataset is a collection of unified diffs of code changes, paired with a short instruction that describes the change. The dataset is derived from the CommitPackFT dataset.",
        reference="https://huggingface.co/datasets/cassanof/CodeEditSearch/viewer",
        dataset={
            "path": "cassanof/CodeEditSearch",
            "revision": "4e51c66e0939303f6928472f13ad0848b2a3f4c0",
        },
        type="Retrieval",
        category="p2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs={lang: [lang + "-Code"] for lang in _LANGS},
        main_score="ndcg_at_10",
        date=("2011-02-12", "2016-01-01"),
        form=["written"],
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="Not specified",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples={
            _EVAL_SPLIT: 1000,
        },
        avg_character_length={"train": 553.50},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        lang_subs = {lang: [] for lang in _LANGS}
        for lang in _LANGS:
            data = datasets.load_dataset(
                split=self._EVAL_SPLIT,
                data_dir=lang,
                **self.metadata_dict["dataset"],
            )
            for row in data:
                lang_subs[lang].append(row)

        self.queries = {}
        self.corpus = {}
        self.relevant_docs = {}

        for lang, sub in lang_subs.items():
            sub = sub[
                : min(len(sub), self.metadata_dict["n_samples"][self._EVAL_SPLIT])
            ]

            self.queries[lang] = {
                self._EVAL_SPLIT: {
                    str(i): row["instruction"] for i, row in enumerate(sub)
                }
            }
            self.corpus[lang] = {
                self._EVAL_SPLIT: {
                    str(row["commit"]): {"text": row["diff"]} for row in sub
                }
            }
            self.relevant_docs[lang] = {
                self._EVAL_SPLIT: {
                    str(i): {row["commit"]: 1} for i, row in enumerate(sub)
                }
            }

        self.data_loaded = True

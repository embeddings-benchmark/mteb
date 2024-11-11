from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "as": ["asm-Beng"],
    "bd": ["brx-Deva"],
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
    "ur": ["urd-Arab"],
}


class IndicSentimentClassification(MultilingualTask, AbsTaskClassification):
    fast_loading = True
    metadata = TaskMetadata(
        name="IndicSentimentClassification",
        dataset={
            "path": "mteb/IndicSentiment",
            "revision": "3389cc78b2ffcbd33639e91dfc57e6b6b6496241",
        },
        description="A new, multilingual, and n-way parallel dataset for sentiment analysis in 13 Indic languages.",
        reference="https://arxiv.org/abs/2212.05409",
        category="s2s",
        modalities=["text"],
        type="Classification",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2022-08-01", "2022-12-20"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
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

    def dataset_transform(self) -> None:
        label_map = {"Negative": 0, "Positive": 1}
        # Convert to standard format
        for lang in self.hf_subsets:
            self.dataset[lang] = self.dataset[lang].filter(
                lambda x: x["LABEL"] is not None
            )
            self.dataset[lang] = self.dataset[lang].rename_columns(
                {"INDIC REVIEW": "text", "LABEL": "label_text"}
            )
            self.dataset[lang] = self.dataset[lang].map(
                lambda x: {"label": label_map[x["label_text"]]}
            )

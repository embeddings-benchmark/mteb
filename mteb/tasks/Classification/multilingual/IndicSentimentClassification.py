from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification, MultilingualTask

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
            "revision": "d522bb117c32f5e0207344f69f7075fc9941168b",
        },
        description="A new, multilingual, and n-way parallel dataset for sentiment analysis in 13 Indic languages.",
        reference="https://arxiv.org/abs/2212.05409",
        category="s2s",
        type="Classification",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2022-08-01", "2022-12-20"),
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="CC0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="machine-translated and verified",
        bibtex_citation="""@article{doddapaneni2022towards,
  title     = {Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages},
  author    = {Sumanth Doddapaneni and Rahul Aralikatte and Gowtham Ramesh and Shreyansh Goyal and Mitesh M. Khapra and Anoop Kunchukuttan and Pratyush Kumar},
  journal   = {Annual Meeting of the Association for Computational Linguistics},
  year      = {2022},
  doi       = {10.18653/v1/2023.acl-long.693}
}""",
        n_samples={"test": 1000},
        avg_character_length={"test": 137.6},
    )

    def dataset_transform(self) -> None:
        label_map = {"Negative": 0, "Positive": 1}
        # Convert to standard format
        for lang in self.langs:
            self.dataset[lang] = self.dataset[lang].filter(
                lambda x: x["LABEL"] is not None
            )
            self.dataset[lang] = self.dataset[lang].rename_columns(
                {"INDIC REVIEW": "text", "LABEL": "label_text"}
            )
            self.dataset[lang] = self.dataset[lang].map(
                lambda x: {"label": label_map[x["label_text"]]}
            )

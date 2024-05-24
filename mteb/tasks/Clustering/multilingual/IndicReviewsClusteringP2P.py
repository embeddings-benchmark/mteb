from __future__ import annotations

from typing import Any

import numpy as np

import datasets
from mteb.abstasks import AbsTaskClustering, MultilingualTask
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


class IndicReviewsClusteringP2P(AbsTaskClustering, MultilingualTask):
    metadata = TaskMetadata(
        name="IndicReviewsClusteringP2P",
        dataset={
            "path": "ai4bharat/IndicSentiment",
            "revision": "ccb472517ce32d103bba9d4f5df121ed5a6592a4",
        },
        description="Clustering of reviews from IndicSentiment dataset. Clustering of 14 sets on the generic categories label.",
        reference="https://arxiv.org/abs/2212.05409",
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
        date=("2022-08-01", "2022-12-20"),
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Thematic clustering"],
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

    def load_data(self, **kwargs: Any) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.hf_subsets:
            self.dataset[lang] = datasets.load_dataset(
                name=f"translation-{lang}",
                **self.metadata_dict["dataset"],
            )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        for lang in self.hf_subsets:
            self.dataset[lang].pop("validation")

            texts = self.dataset[lang]["test"]["INDIC REVIEW"]
            labels = self.dataset[lang]["test"]["GENERIC CATEGORIES"]

            new_format = {
                "sentences": [split.tolist() for split in np.array_split(texts, 5)],
                "labels": [split.tolist() for split in np.array_split(labels, 5)],
            }
            self.dataset[lang]["test"] = datasets.Dataset.from_dict(new_format)

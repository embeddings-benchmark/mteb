from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
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


class IndicReviewsClusteringP2P(AbsTaskClustering, MultilingualTask):
    metadata = TaskMetadata(
        name="IndicReviewsClusteringP2P",
        dataset={
            "path": "mteb/IndicReviewsClusteringP2P",
            "revision": "add94d3b9154cc561bbad0e16ee66ebf5941f8a4",
        },
        description="Clustering of reviews from IndicSentiment dataset. Clustering of 14 sets on the generic categories label.",
        reference="https://arxiv.org/abs/2212.05409",
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
        date=("2022-08-01", "2022-12-20"),
        domains=["Reviews", "Written"],
        task_subtypes=["Thematic clustering"],
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

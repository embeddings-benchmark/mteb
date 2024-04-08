from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class RedditClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="RedditClusteringP2P",
        description="Clustering of title+posts from reddit. Clustering of 10 sets of 50k paragraphs and 40 sets of 10k paragraphs.",
        reference="https://arxiv.org/abs/2104.07081",
        dataset={
            "path": "mteb/reddit-clustering-p2p",
            "revision": "385e3cb46b4cfa89021f56c4380204149d0efe33",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="v_measure",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 459399},
        avg_character_length={"test": 727.7},
    )

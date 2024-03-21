from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class RedditClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="RedditClusteringP2P",
        description="Clustering of title+posts from reddit. Clustering of 10 sets of 50k paragraphs and 40 sets of 10k paragraphs.",
        reference="https://arxiv.org/abs/2104.07081",
        hf_hub_name="mteb/reddit-clustering-p2p",
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="v_measure",
        revision="24640382cdbf8abc73003fb0fa6d111a705499eb",
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
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class RedditClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="RedditClustering",
        description="Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.",
        reference="https://arxiv.org/abs/2104.07081",
        hf_hub_name="mteb/reddit-clustering",
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="v_measure",
        revision="24640382cdbf8abc73003fb0fa6d111a705499eb",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license="",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class EightTagsClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="EightTagsClustering",
        description="Clustering of headlines from social media posts in Polish belonging to 8 categories: film, history, "
        "food, medicine, motorization, work, sport and technology.",
        reference="https://aclanthology.org/2020.lrec-1.207.pdf",
        hf_hub_name="mteb/polish-clustering",
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["pl"],
        main_score="v_measure",
        revision="e7a26af6f3ae46b30dde8737f02c07b1505bcc73",
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

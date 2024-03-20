from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class ArxivClusteringS2S(AbsTaskClustering):
    metadata = TaskMetadata(
        name="ArxivClusteringS2S",
        description="Clustering of titles from arxiv. Clustering of 30 sets, either on the main or secondary category",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        hf_hub_name="mteb/arxiv-clustering-s2s",
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="v_measure",
        revision="f910caf1a6075f7329cdf8c1a6135696f37dbd53",
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

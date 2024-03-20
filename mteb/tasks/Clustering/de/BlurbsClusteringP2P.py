from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class BlurbsClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="BlurbsClusteringP2P",
        description="Clustering of book titles+blurbs. Clustering of 28 sets, either on the main or secondary genre.",
        reference="https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html",
        hf_hub_name="slvnwhrl/blurbs-clustering-p2p",
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["de"],
        main_score="v_measure",
        revision="a2dd5b02a77de3466a3eaa98ae586b5610314496",
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

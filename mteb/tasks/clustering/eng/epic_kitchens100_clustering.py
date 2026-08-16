from __future__ import annotations

from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class EPICKitchens100Clustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="EPICKitchens100Clustering",
        description=(
            "Clustering of egocentric kitchen-activity video clips into 25 verb "
            "categories from EPIC-KITCHENS-100. 500 clips, exactly 20 per class, "
            "sampled from lightly-ai's pre-trimmed action-clip mirror (each clip "
            "corresponds to one official narration annotation); the class label "
            "is the most frequent surface form of each verb_class, since "
            "individual narrations use varied verb strings for the same class "
            "(e.g. both 'pour-out' and 'pour-from' fall under one verb_class)."
        ),
        reference="https://huggingface.co/datasets/lightly-ai/epic-kitchens-100-clips",
        dataset={
            "path": "yaswanth169/EPIC-KITCHENS100-VideoClustering",
            "revision": "dcce15464c759cb96ca5f8590e802ec2d0d5531e",
        },
        type="VideoClustering",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2018-01-01", "2021-01-01"),
        domains=["Activity", "Scene"],
        task_subtypes=["Activity recognition"],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=r"""
@article{Damen2021RESCALING,
  author = {Damen, Dima and Doughty, Hazel and Farinella, Giovanni Maria and Furnari, Antonino and Ma, Jian and Kazakos, Evangelos and Moltisanti, Davide and Munro, Jonathan and Perrett, Toby and Price, Will and Wray, Michael},
  journal = {International Journal of Computer Vision (IJCV)},
  title = {Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100},
  volume = {130},
  year = {2022},
}
""",
        is_beta=True,
    )
    max_fraction_of_documents_to_embed = None
    input_column_name: str = "video"
    label_column_name: str = "label"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        for split in self.metadata.eval_splits:
            self.dataset[split] = self.dataset[split].select_columns(
                ["video", "label"],
            )

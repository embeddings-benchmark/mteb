from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import (
    AbsTaskZeroShotClassification,
)


class HMDB51ZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="HMDB51ZeroShot",
        description="HMDB51 is a large video database for human motion recognition with 51 action categories from digitized movies and online sources. Used official split 1 across 51 action classes (~3,570 train / ~1,530 test).",
        reference="https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/",
        dataset={
            "path": "mteb/HMDB51",
            "revision": "7f9af5438a855e9348fb23ecb5ec740a9c21daf3",
        },
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2011-01-01",
            "2011-12-31",
        ),
        domains=["Scene", "Web"],
        task_subtypes=["Activity recognition"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{6126543,
  author = {Kuehne, H. and Jhuang, H. and Garrote, E. and Poggio, T. and Serre, T.},
  booktitle = {2011 International Conference on Computer Vision},
  doi = {10.1109/ICCV.2011.6126543},
  keywords = {Cameras;YouTube;Databases;Training;Visualization;Humans;Motion pictures},
  number = {},
  pages = {2556-2563},
  title = {HMDB: A large video database for human motion recognition},
  volume = {},
  year = {2011},
}
""",
        is_beta=True,
    )

    input_column_name: str = "video"

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a video of {name.replace('_', ' ')}"
            for name in self.dataset["test"].features[self.label_column_name].names
        ]

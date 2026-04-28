from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import (
    AbsTaskZeroShotClassification,
)


class HMDB51ZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="HMDB51ZeroShot",
        description="HMDB51 is a large video database for human motion recognition with 51 action categories from digitized movies and online sources.",
        reference="https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/",
        dataset={
            "path": "mteb/HMDB51",
            "revision": "73e5ac9cd9536c406d0046f3d6046785885f7ebe",
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
        domains=["Scene"],
        task_subtypes=["Activity recognition"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
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

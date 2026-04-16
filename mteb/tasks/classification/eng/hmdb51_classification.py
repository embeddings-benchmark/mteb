from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

# Train rows whose embedded video bytes fail torchcodec decode on revision below (Hub parquet).
_BAD_VIDEO_TRAIN_INDICES: frozenset[int] = frozenset({2555, 3476})


class HMDB51Classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HMDB51Classification",
        description="HMDB51 is a large video database for human motion recognition with 51 action categories from digitized movies and online sources.",
        reference="https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/",
        dataset={
            "path": "mteb/HMDB51",
            "revision": "73e5ac9cd9536c406d0046f3d6046785885f7ebe",
        },
        type="VideoClassification",
        category="v2c",
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
        is_beta=True,
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
    )

    input_column_name = "video"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        if self.metadata.dataset["revision"] != "73e5ac9cd9536c406d0046f3d6046785885f7ebe":
            return
        train = self.dataset["train"]
        if len(train) != 3570:
            return
        keep = [i for i in range(len(train)) if i not in _BAD_VIDEO_TRAIN_INDICES]
        self.dataset["train"] = train.select(keep)

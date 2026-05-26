from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class HMDB51Classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HMDB51Classification",
        description="HMDB51 is a large video database for human motion recognition with 51 action categories from digitized movies and online sources. Used official split 1 across 51 action classes (~3,570 train / ~1,530 test).",
        reference="https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/",
        dataset={
            "path": "mteb/HMDB51",
            "revision": "7f9af5438a855e9348fb23ecb5ec740a9c21daf3",
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
        domains=["Scene", "Web"],
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

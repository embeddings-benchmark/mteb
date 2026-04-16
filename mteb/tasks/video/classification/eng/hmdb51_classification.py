from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="va2c",
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
@INPROCEEDINGS{6126543,
  author={Kuehne, H. and Jhuang, H. and Garrote, E. and Poggio, T. and Serre, T.},
  booktitle={2011 International Conference on Computer Vision}, 
  title={HMDB: A large video database for human motion recognition}, 
  year={2011},
  volume={},
  number={},
  pages={2556-2563},
  keywords={Cameras;YouTube;Databases;Training;Visualization;Humans;Motion pictures},
  doi={10.1109/ICCV.2011.6126543}}

""",
    )

    input_column_name = "video"
    label_column_name: str = "label"

    is_cross_validation: bool = False

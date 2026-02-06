from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class TAUAcousticScenes2022Mobile(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TAUAcousticScenes2022Mobile",
        description="TAU Urban Acoustic Scenes 2022 Mobile, development dataset consists of 1-second audio recordings from 12 European cities in 10 different acoustic scenes using 4 different devices. This is a stratified subsampled version of the evaluation_setup subset of the original dataset.",
        reference="https://zenodo.org/records/6337421",
        dataset={
            "path": "mteb/tau-acoustic-scenes-2022-mobile-mini",
            "revision": "d0da0ed80d22944c7a5690c4b570683d45c4dfaf",
        },
        type="AudioClassification",
        category="a2c",
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="accuracy",
        date=("2022-03-08", "2022-03-08"),
        domains=[
            "AudioScene",
        ],
        task_subtypes=["Environment Sound Classification"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@dataset{heittola_2022_6337421,
  author = {Toni Heittola and Annamaria Mesaros and Tuomas Virtanen},
  publisher = {Zenodo},
  title = {TAU Urban Acoustic Scenes 2022 Mobile, Development Dataset},
  url = {https://doi.org/10.5281/zenodo.6337421},
  year = {2022},
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "scene_label"

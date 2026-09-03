from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class TAUAcousticScenes2022MobileClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="TAUAcousticScenes2022MobileClustering",
        description="Clustering task over TAU Urban Acoustic Scenes 2022 Mobile, 1-second recordings made in 12 European cities across 10 acoustic scenes with 4 different devices. mteb already scores this dataset with labels through TAUAcousticScenes2022Mobile; this asks instead whether audio embeddings separate urban acoustic environments with no labels at all. Same stratified subsample of the evaluation_setup subset, at the same revision, so the two tasks are directly comparable.",
        reference="https://zenodo.org/records/6337421",
        dataset={
            "path": "mteb/tau-acoustic-scenes-2022-mobile-mini",
            "revision": "d0da0ed80d22944c7a5690c4b570683d45c4dfaf",
        },
        type="AudioClustering",
        category="a2a",
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="v_measure",
        date=("2022-03-08", "2022-03-08"),
        domains=[
            "AudioScene",
        ],
        task_subtypes=["Environment Sound Clustering"],
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

    max_fraction_of_documents_to_embed = None
    input_column_name: str = "audio"
    label_column_name: str = "scene_label"

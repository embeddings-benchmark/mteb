from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class TAUAcousticScenes2022MobileClustering(AbsTaskClustering):
    label_column_name: str = "scene_label"
    input_column_name: str = "audio"
    max_fraction_of_documents_to_embed = None
    metadata = TaskMetadata(
        name="TAUAcousticScenes2022MobileClustering",
        description=(
            "Audio clustering of urban acoustic scenes from the TAU Urban Acoustic "
            "Scenes 2022 Mobile dataset. Contains 1-second recordings from 12 European "
            "cities captured by 4 mobile devices across 10 scene types (airport, bus, "
            "metro, metro station, park, public square, shopping mall, street "
            "pedestrian, street traffic, tram). Evaluates whether audio embeddings "
            "can distinguish urban acoustic environments."
        ),
        reference="https://zenodo.org/records/6337421",
        dataset={
            "path": "mteb/tau-acoustic-scenes-2022-mobile-mini",
            "revision": "d0da0ed80d22944c7a5690c4b570683d45c4dfaf",
        },
        type="AudioClustering",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="v_measure",
        date=("2022-03-08", "2022-03-08"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Clustering"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
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

from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class RAVDESSAVClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="RAVDESS_AVClustering",
        description="Emotion clustering task with video and audio data for 8 emotions: neutral, calm, happy, sad, angry, fearful, surprise, and disgust expressions.",
        reference="https://doi.org/10.1371/journal.pone.0196391",
        dataset={
            "path": "mteb/RAVDESS_AV",
            "revision": "13af08387c3ce5e86c179a3718eb158669268d65",
        },
        type="VideoClustering",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2018-03-01", "2018-03-16"),
        domains=["Spoken", "Scene"],
        task_subtypes=["Emotion Clustering"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=r"""
@article{10.1371/journal.pone.0196391,
  author = {Livingstone, Steven R. AND Russo, Frank A.},
  doi = {10.1371/journal.pone.0196391},
  journal = {PLOS ONE},
  month = {05},
  number = {5},
  pages = {1-35},
  publisher = {Public Library of Science},
  title = {The Ryerson Audio-Visual Database ofal Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English},
  url = {https://doi.org/10.1371/journal.pone.0196391},
  volume = {13},
  year = {2018},
}
""",
    )
    max_fraction_of_documents_to_embed = None
    input_column_name: str = "video"
    label_column_name: str = "emotion"

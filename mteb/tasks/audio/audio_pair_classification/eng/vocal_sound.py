from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class VocalSoundPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="VocalSoundPairClassification",
        description="Recognizing whether two audio clips are the same human vocal expression (laughing, sighing, etc.)",
        reference="https://www.researchgate.net/publication/360793875_Vocalsound_A_Dataset_for_Improving_Human_Vocal_Sounds_Recognition/citations",
        dataset={
            "path": "mteb/VocalSoundPairClassification",
            "revision": "f166121e5e0a71f308f63ec0d610ea0e753de996",
        },
        type="AudioPairClassification",
        category="a2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2014-01-01", "2014-12-31"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        annotations_creators="human-annotated",
        dialect=[],
        license="cc-by-sa-4.0",
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{inproceedings,
  author = {Gong, Yuan and Yu, Jin and Glass, James},
  doi = {10.1109/ICASSP43922.2022.9746828},
  month = {05},
  pages = {151-155},
  title = {Vocalsound: A Dataset for Improving Human Vocal Sounds Recognition},
  year = {2022},
}
""",
    )

    input1_column_name: str = "audio1"
    input2_column_name: str = "audio2"
    label_column_name: str = "label"

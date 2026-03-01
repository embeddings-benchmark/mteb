from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class CREMAD(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CREMA_D",
        description="Emotion classification of audio into one of 6 classes: Anger, Disgust, Fear, Happy, Neutral, Sad.",
        reference="https://huggingface.co/datasets/silky1708/CREMA-D",
        dataset={
            "path": "mteb/crema-d",
            "revision": "a9d0821955c418752248b8310438854e56a1cf34",
        },
        type="AudioClassification",
        category="a2c",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2014-01-01", "2014-12-31"),
        domains=["Speech"],
        task_subtypes=["Emotion classification"],
        license="http://opendatacommons.org/licenses/odbl/1.0/",  # Open Database License
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation=r"""
@article{Cao2014-ih,
  author = {Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur,
Ruben C and Nenkova, Ani and Verma, Ragini},
  copyright = {https://ieeexplore.ieee.org/Xplorehelp/downloads/license-information/IEEE.html},
  journal = {IEEE Transactions on Affective Computing},
  keywords = {Emotional corpora; facial expression; multi-modal recognition;
voice expression},
  language = {en},
  month = oct,
  number = {4},
  pages = {377--390},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  title = {{CREMA-D}: Crowd-sourced emotional multimodal actors dataset},
  volume = {5},
  year = {2014},
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"

    is_cross_validation: bool = True

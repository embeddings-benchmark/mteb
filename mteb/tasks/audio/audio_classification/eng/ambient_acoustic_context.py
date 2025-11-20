from mteb.abstasks.audio.abs_task_audio_classification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class AmbientAcousticContextClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="AmbientAcousticContext",
        description="The Ambient Acoustic Context dataset contains 1-second segments for activities that occur in a workplace setting. This is a downsampled version with ~100 train and ~50 test samples per class.",
        reference="https://dl.acm.org/doi/10.1145/3379503.3403535",
        dataset={
            "path": "mteb/ambient-acoustic-context-small",
            "revision": "8f4de158d4162de768ebb4dc0594429d785077da",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2020-12-31"),  # Paper publication date
        domains=["Spoken", "Speech"],
        task_subtypes=["Environment Sound Classification"],
        license="not specified",  # Not specified in dataset card
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{10.1145/3379503.3403535,
  address = {New York, NY, USA},
  articleno = {33},
  author = {Park, Chunjong and Min, Chulhong and Bhattacharya, Sourav and Kawsar, Fahim},
  booktitle = {22nd International Conference on Human-Computer Interaction with Mobile Devices and Services},
  doi = {10.1145/3379503.3403535},
  isbn = {9781450375160},
  keywords = {Acoustic ambient context, Conversational agents},
  location = {Oldenburg, Germany},
  numpages = {9},
  publisher = {Association for Computing Machinery},
  series = {MobileHCI '20},
  title = {Augmenting Conversational Agents with Ambient Acoustic Contexts},
  url = {https://doi.org/10.1145/3379503.3403535},
  year = {2020},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = None  # Not needed as dataset is already balanced
    is_cross_validation: bool = False

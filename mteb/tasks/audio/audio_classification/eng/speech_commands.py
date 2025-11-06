from mteb.abstasks.audio.abs_task_audio_classification import (
    AbsTaskAudioClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class SpeechCommandsClassification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="SpeechCommands",
        description="A set of one-second .wav audio files, each containing a single spoken English word or background noise. To keep evaluation fast, we use a downsampled version of the original dataset by keeping ~50 samples per class for training.",
        reference="https://arxiv.org/abs/1804.03209",
        dataset={
            "path": "mteb/speech-commands-mini",
            "revision": "3ac713aa0829eeadda73182f38bbbd788d21254b",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-04-11", "2018-04-11"),  # v0.02 release date
        domains=["Speech"],
        task_subtypes=["Spoken Language Identification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@article{speechcommands2018,
  author = {Pete Warden},
  journal = {arXiv preprint arXiv:1804.03209},
  title = {Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition},
  year = {2018},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "label"
    samples_per_label: int = 50
    is_cross_validation: bool = False

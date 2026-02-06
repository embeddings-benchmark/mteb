from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SpeechCommandsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SpeechCommands",
        description="A set of one-second .wav audio files, each containing a single spoken English word or background noise. To keep evaluation fast, we use a downsampled version of the original dataset by keeping ~50 samples per class for training.",
        reference="https://arxiv.org/abs/1804.03209",
        dataset={
            "path": "mteb/speech-commands-mini",
            "revision": "3ac713aa0829eeadda73182f38bbbd788d21254b",
        },
        type="AudioClassification",
        category="a2c",
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
@article{DBLP:journals/corr/abs-1804-03209,
  author = {Pete Warden},
  bibsource = {dblp computer science bibliography, https://dblp.org},
  biburl = {https://dblp.org/rec/journals/corr/abs-1804-03209.bib},
  eprint = {1804.03209},
  eprinttype = {arXiv},
  journal = {CoRR},
  timestamp = {Mon, 13 Aug 2018 16:48:32 +0200},
  title = {Speech Commands: {A} Dataset for Limited-Vocabulary Speech Recognition},
  url = {http://arxiv.org/abs/1804.03209},
  volume = {abs/1804.03209},
  year = {2018},
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"

    is_cross_validation: bool = False
